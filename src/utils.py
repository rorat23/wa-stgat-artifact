import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler


def prepare_dataloaders(config):
    """
    Loads preprocessed tensors, scales features, and constructs PyG batched dataloaders.
    """
    try:
        tier1_df = pd.read_parquet(config['paths']['processed_tensor'])
        edge_index = torch.load(config['paths']['edge_index'], weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError("Processed data not found. Run scripts/preprocess.py first.")


    feature_cols = ['hour', 'minute', 'day_of_week', 'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4']
    weather_cols = ['temperature', 'rain_intensity']


    scaler_features = StandardScaler()
    scaler_target = StandardScaler()


    tier1_df[feature_cols] = scaler_features.fit_transform(tier1_df[feature_cols])
    tier1_df['demand_scaled'] = scaler_target.fit_transform(tier1_df[['demand']])


    grouped = tier1_df.groupby('time_bin')
    data_list = []


    for t, snapshot in grouped:
        snapshot = snapshot.sort_values('PULocationID')
        if len(snapshot) != config['model']['num_nodes']:
            continue
            
        x = torch.tensor(snapshot[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(snapshot['demand_scaled'].values, dtype=torch.float32).unsqueeze(1)
        y_raw = torch.tensor(snapshot['demand'].values, dtype=torch.float32).unsqueeze(1)
        w = torch.tensor(snapshot[weather_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0) 
        
        data = Data(x=x, edge_index=edge_index, y=y, weather=w, y_raw=y_raw)
        data_list.append(data)


    train_size = int(0.7 * len(data_list))
    val_size = int(0.15 * len(data_list))


    train_dataset = data_list[:train_size]
    val_dataset = data_list[train_size : train_size + val_size]
    test_dataset = data_list[train_size + val_size :]


    bs = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


    return train_loader, val_loader, test_loader, scaler_target


