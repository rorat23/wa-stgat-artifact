import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.utils import softmax


class WeatherGatedAttentionConvBatched(MessagePassing):
    def __init__(self, node_in_channels, weather_in_channels, out_channels):
        super(WeatherGatedAttentionConvBatched, self).__init__(aggr='add')
        self.lin_node = nn.Linear(node_in_channels, out_channels, bias=False)
        self.lin_weather = nn.Linear(weather_in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 3 * out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_weather.weight)
        nn.init.xavier_uniform_(self.att)


    def forward(self, x, edge_index, weather_vec, batch_idx):
        x_transformed = self.lin_node(x)
        w_embedded = self.lin_weather(weather_vec)[batch_idx] 
        return self.propagate(edge_index, x=x_transformed, w_embedded=w_embedded)


    def message(self, edge_index_i, x_i, x_j, w_embedded_i, index, ptr, size_i):
        concat_features = torch.cat([x_i, x_j, w_embedded_i], dim=-1)
        e_ij = self.leaky_relu((concat_features * self.att).sum(dim=-1))
        alpha = softmax(e_ij, index, ptr, size_i)
        return x_j * alpha.view(-1, 1)


class Tier1_SurgePredictor(nn.Module):
    def __init__(self, num_nodes, node_features, weather_features, hidden_dim):
        super(Tier1_SurgePredictor, self).__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.gat = WeatherGatedAttentionConvBatched(node_features + hidden_dim, weather_features, hidden_dim)
        self.skip = nn.Linear(node_features + hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)


    def forward(self, x, edge_index, weather_vec, batch_idx):
        local_node_ids = (torch.arange(x.size(0), device=x.device) % 263)
        emb = self.node_emb(local_node_ids)
        x_combined = torch.cat([x, emb], dim=-1)
        spatial_dynamic = F.relu(self.gat(x_combined, edge_index, weather_vec, batch_idx))
        identity = F.relu(self.skip(x_combined))
        return self.regressor(spatial_dynamic + identity)


# Ablation Baselines
class STGCN_Baseline(nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim):
        super(STGCN_Baseline, self).__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.gcn = GCNConv(node_features + hidden_dim, hidden_dim)
        self.skip = nn.Linear(node_features + hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)


    def forward(self, x, edge_index):
        node_ids = torch.arange(x.size(0), device=x.device)
        emb = self.node_emb(node_ids)
        x_combined = torch.cat([x, emb], dim=-1)
        return self.regressor(F.relu(self.gcn(x_combined, edge_index)) + F.relu(self.skip(x_combined)))


