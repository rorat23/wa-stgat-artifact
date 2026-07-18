import os
import yaml
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import WA_STGAT, Baseline_WeatherConcat, Ablation_NoSkip, Ablation_NoSpatialEmb
from src.utils import prepare_dataloaders

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, _, scaler_target = prepare_dataloaders(config)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 1. Initializing Leaderboard Training on {device} ---")

    # Define the leaderboard architectures
    models = {
        'wa_stgat': WA_STGAT,
        'baseline_concat': Baseline_WeatherConcat,
        'ablation_noskip': Ablation_NoSkip,
        'ablation_noemb': Ablation_NoSpatialEmb
    }

    os.makedirs('results', exist_ok=True)

    for model_name, ModelClass in models.items():
        print(f"\n>> Training: {model_name.upper()}")

        model = ModelClass(
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

        for epoch in range(config['training']['epochs']):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred_scaled = model(batch.x, batch.edge_index, batch.weather, batch.batch)
                loss = criterion(pred_scaled, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred_scaled = model(batch.x, batch.edge_index, batch.weather, batch.batch)
                    pred_raw = torch.tensor(scaler_target.inverse_transform(pred_scaled.cpu().numpy())).to(device)
                    val_loss += criterion(pred_raw, batch.y_raw).item() * batch.num_graphs

            avg_val_rmse = (val_loss / len(val_loader.dataset)) ** 0.5

            # Silent early stopping for clean logs
            if avg_val_rmse < best_val_loss:
                best_val_loss = avg_val_rmse
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['training']['patience']:
                    print(f"   [ Early stopping triggered at Epoch {epoch+1} | Best RMSE: {best_val_loss:.3f} ]")
                    break

        torch.save(best_weights, f"results/{model_name}_weights.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    train(args.config)
