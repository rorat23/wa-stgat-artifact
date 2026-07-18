import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.utils import softmax

# ---------------------------------------------------------
# 1. FIXED: WEATHER-CONDITIONED DYNAMIC ATTENTION
# ---------------------------------------------------------
class WeatherConditionedGATConv(MessagePassing):
    def __init__(self, node_in_channels, weather_in_channels, out_channels):
        super(WeatherConditionedGATConv, self).__init__(aggr='add')
        self.lin_node = nn.Linear(node_in_channels, out_channels, bias=False)
        self.lin_weather = nn.Linear(weather_in_channels, out_channels, bias=False)

        # The attention vector now expects 4 components:
        # [node_i, node_j, node_i_gated, node_j_gated] -> 4 * out_channels
        self.att = nn.Parameter(torch.Tensor(1, 4 * out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_weather.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        # 1. Project nodes and weather into the same latent dimension
        h = self.lin_node(x)
        w_emb = self.lin_weather(weather_vec)

        # 2. Broadcast the global weather vector to all nodes in their respective graphs
        w_emb_broadcast = w_emb[batch_idx]

        return self.propagate(edge_index, h=h, w_emb=w_emb_broadcast)

    def message(self, edge_index_i, h_i, h_j, w_emb_i, w_emb_j, index, ptr, size_i):
        # 3. CRITICAL FIX: Hadamard Product (Element-wise multiplication)
        # The weather vector dynamically scales the hidden features of the nodes
        weather_gated_i = h_i * w_emb_i
        weather_gated_j = h_j * w_emb_j

        # 4. Concatenate base features WITH the weather-gated features
        concat_features = torch.cat([h_i, h_j, weather_gated_i, weather_gated_j], dim=-1)

        # 5. Calculate attention weights
        e_ij = self.leaky_relu((concat_features * self.att).sum(dim=-1))
        alpha = softmax(e_ij, index, ptr, size_i)

        # 6. Pass the weather-modulated message scaled by the attention weight
        return weather_gated_j * alpha.view(-1, 1)

# ---------------------------------------------------------
# 2. THE MAIN WA-STGAT ARCHITECTURE
# ---------------------------------------------------------
class Tier1_SurgePredictor(nn.Module):
    def __init__(self, num_nodes, node_features, weather_features, hidden_dim):
        super(Tier1_SurgePredictor, self).__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        # Call the newly fixed attention mechanism
        self.gat = WeatherConditionedGATConv(node_features + hidden_dim, weather_features, hidden_dim)
        self.skip = nn.Linear(node_features + hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        local_node_ids = (torch.arange(x.size(0), device=x.device) % 263)
        emb = self.node_emb(local_node_ids)

        x_combined = torch.cat([x, emb], dim=-1)
        spatial_dynamic = F.relu(self.gat(x_combined, edge_index, weather_vec, batch_idx))
        identity = F.relu(self.skip(x_combined))

        return self.regressor(spatial_dynamic + identity)

# ---------------------------------------------------------
# 3. ABLATION BASELINES (ST-GCN dropped due to convergence failure)
# ---------------------------------------------------------
class Ablation_NoWeather(nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.gat = GATConv(node_features + hidden_dim, hidden_dim)
        self.skip = nn.Linear(node_features + hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        local_node_ids = (torch.arange(x.size(0), device=x.device) % 263)
        emb = self.node_emb(local_node_ids)
        x_combined = torch.cat([x, emb], dim=-1)
        return self.regressor(F.relu(self.gat(x_combined, edge_index)) + F.relu(self.skip(x_combined)))

class Ablation_NoEmb(nn.Module):
    def __init__(self, node_features, weather_features, hidden_dim):
        super().__init__()
        self.gat = WeatherConditionedGATConv(node_features, weather_features, hidden_dim)
        self.skip = nn.Linear(node_features, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        return self.regressor(F.relu(self.gat(x, edge_index, weather_vec, batch_idx)) + F.relu(self.skip(x)))

class Ablation_NoSkip(nn.Module):
    def __init__(self, num_nodes, node_features, weather_features, hidden_dim):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.gat = WeatherConditionedGATConv(node_features + hidden_dim, weather_features, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        local_node_ids = (torch.arange(x.size(0), device=x.device) % 263)
        emb = self.node_emb(local_node_ids)
        x_combined = torch.cat([x, emb], dim=-1)
        return self.regressor(F.relu(self.gat(x_combined, edge_index, weather_vec, batch_idx)))
