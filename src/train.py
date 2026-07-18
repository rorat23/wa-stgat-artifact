import os
import yaml
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim


from src.model import Tier1_SurgePredictor
from src.utils import prepare_dataloaders


def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    print("--- 1. Loading Configured Data ---")
    train_loader, val_loader, _, scaler_target = prepare_dataloaders(config)


    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- 2. Initializing Model on {device} ---")
    
    model = Tier1_SurgePredictor(
        num_nodes=config['model']['num_nodes'],
        node_features=config['model']['node_features'],
        weather_features=config['model']['weather_features'],
        hidden_dim=config['model']['hidden_dim']
    ).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])


    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    start_time = time.time()


    print("\n--- 3. Commencing Batched Training ---")
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_scaled = model(batch.x, batch.edge_index, batch.weather, batch.batch)
            loss = criterion(pred_scaled, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs


        avg_train_loss = (train_loss / len(train_loader.dataset)) ** 0.5


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_scaled = model(batch.x, batch.edge_index, batch.weather, batch.batch)
                pred_raw = torch.tensor(scaler_target.inverse_transform(pred_scaled.cpu().numpy())).to(device)
                val_loss += criterion(pred_raw, batch.y_raw).item() * batch.num_graphs


        avg_val_rmse = (val_loss / len(val_loader.dataset)) ** 0.5
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val RMSE (Raw): {avg_val_rmse:.4f}")


        if avg_val_rmse < best_val_loss:
            best_val_loss = avg_val_rmse
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"[ Early stopping triggered at Epoch {epoch+1} ]")
                break


    print(f"\n--- Training Complete in {(time.time() - start_time) / 60:.2f} minutes ---")
    os.makedirs(os.path.dirname(config['paths']['checkpoint']), exist_ok=True)
    torch.save(best_weights, config['paths']['checkpoint'])
    print(f"Model saved to {config['paths']['checkpoint']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    train(args.config)


