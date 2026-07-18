import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# ---------------------------------------------------------
# 1. CORE INNOVATION: Weather-Conditioned Hadamard Attention
# ---------------------------------------------------------
class WeatherConditionedGATConv(MessagePassing):
    def __init__(self, node_in_channels, weather_in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(node_in_channels, out_channels, bias=False)
        self.lin_weather = nn.Linear(weather_in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 4 * out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_weather.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        h = self.lin_node(x)
        w_emb = self.lin_weather(weather_vec)[batch_idx]
        return self.propagate(edge_index, h=h, w_emb=w_emb)

    def message(self, edge_index_i, h_i, h_j, w_emb_i, w_emb_j, index, ptr, size_i):
        wg_i = h_i * w_emb_i  # Hadamard Gate
        wg_j = h_j * w_emb_j  # Hadamard Gate
        concat = torch.cat([h_i, h_j, wg_i, wg_j], dim=-1)
        e_ij = self.leaky_relu((concat * self.att).sum(dim=-1))
        alpha = softmax(e_ij, index, ptr, size_i)
        return wg_j * alpha.view(-1, 1)

# ---------------------------------------------------------
# 2. BASELINE: Simple Weather-Concatenation Attention
# ---------------------------------------------------------
class WeatherConcatGATConv(MessagePassing):
    def __init__(self, node_in_channels, weather_in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(node_in_channels, out_channels, bias=False)
        self.lin_weather = nn.Linear(weather_in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 3 * out_channels)) # Only 3 components
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_weather.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        h = self.lin_node(x)
        w_emb = self.lin_weather(weather_vec)[batch_idx]
        return self.propagate(edge_index, h=h, w_emb=w_emb)

    def message(self, edge_index_i, h_i, h_j, w_emb_i, index, ptr, size_i):
        concat = torch.cat([h_i, h_j, w_emb_i], dim=-1) # Pure concatenation, no gating
        e_ij = self.leaky_relu((concat * self.att).sum(dim=-1))
        alpha = softmax(e_ij, index, ptr, size_i)
        return h_j * alpha.view(-1, 1)

# ---------------------------------------------------------
# 3. THE MODELS DICTIONARY EXPORTS
# ---------------------------------------------------------
class WA_STGAT(nn.Module):
    def __init__(self, num_nodes, node_features, weather_features, hidden_dim):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.gat = WeatherConditionedGATConv(node_features + hidden_dim, weather_features, hidden_dim)
        self.skip = nn.Linear(node_features + hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        local_node_ids = (torch.arange(x.size(0), device=x.device) % 263)
        emb = self.node_emb(local_node_ids)
        x_combined = torch.cat([x, emb], dim=-1)
        return self.regressor(F.relu(self.gat(x_combined, edge_index, weather_vec, batch_idx)) + F.relu(self.skip(x_combined)))

class Baseline_WeatherConcat(nn.Module):
    def __init__(self, num_nodes, node_features, weather_features, hidden_dim):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.gat = WeatherConcatGATConv(node_features + hidden_dim, weather_features, hidden_dim)
        self.skip = nn.Linear(node_features + hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        local_node_ids = (torch.arange(x.size(0), device=x.device) % 263)
        emb = self.node_emb(local_node_ids)
        x_combined = torch.cat([x, emb], dim=-1)
        return self.regressor(F.relu(self.gat(x_combined, edge_index, weather_vec, batch_idx)) + F.relu(self.skip(x_combined)))

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

class Ablation_NoSpatialEmb(nn.Module):
    def __init__(self, num_nodes, node_features, weather_features, hidden_dim):
        super().__init__()
        self.gat = WeatherConditionedGATConv(node_features, weather_features, hidden_dim)
        self.skip = nn.Linear(node_features, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, weather_vec, batch_idx):
        return self.regressor(F.relu(self.gat(x, edge_index, weather_vec, batch_idx)) + F.relu(self.skip(x)))
