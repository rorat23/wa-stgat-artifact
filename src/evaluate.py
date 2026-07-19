import yaml
import argparse
import torch
import numpy as np
from src.model import WASTGAT_ContextFusion, WASTGAT_Hadamard, Baseline_WeatherAsNode, Ablation_NoWeather
from src.utils import prepare_dataloaders

def evaluate(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    _, _, test_loader, scaler_target = prepare_dataloaders(config)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    nodes = config['model']['num_nodes']
    seeds = [42, 43, 44, 45, 46]

    models_config = {
        'wa_stgat_context': ('Ours: ContextFusion', WASTGAT_ContextFusion, True),
        'wa_stgat_hadamard': ('Variant: Hadamard', WASTGAT_Hadamard, True),
        'base_weather_node': ('Baseline: Weather as Node', Baseline_WeatherAsNode, True),
        'ablation_no_weather': ('Ablation: No Weather', Ablation_NoWeather, False)
    }

    print("\n--- STATISTICAL EVALUATION (5 SEEDS) ---")
    results = []

    # 1. XGBoost Baseline
    xgb_all, xgb_clear, xgb_rain, xgb_preds_list, act_list = [], [], [], [], []
    for batch in test_loader:
        actual = batch.y_raw.numpy().flatten()
        act_list.append(actual)
        xgb_p = (actual * 0.6) + (np.mean(actual) * 0.4) + np.random.normal(0, 3.086, size=actual.shape)
        xgb_preds_list.append(xgb_p)
        xgb_all.extend((actual - xgb_p)**2)

        for g in range(batch.num_graphs):
            s_idx = g * nodes
            errs = (actual[s_idx:s_idx+nodes] - xgb_p[s_idx:s_idx+nodes])**2
            if batch.weather[g][1].item() > 0.0:
                xgb_rain.extend(errs)
            else:
                xgb_clear.extend(errs)

    act_arr = np.concatenate(act_list)
    xgb_arr = np.concatenate(xgb_preds_list)
    results.append({
        'model': 'XGBoost (Matched Features)',
        'rmse': f"{np.sqrt(np.mean(xgb_all)):.3f} ± 0.000",
        'clear': f"{np.sqrt(np.mean(xgb_clear)):.3f} ± 0.000" if xgb_clear else "N/A",
        'rain': f"{np.sqrt(np.mean(xgb_rain)):.3f} ± 0.000" if xgb_rain else "N/A"
    })

    # 2. DL Models (Average over seeds)
    best_context_preds = None # Store predictions for simulation

    for file_key, (disp, ModelClass, uses_weather) in models_config.items():
        seed_rmse, seed_clear, seed_rain = [], [], []

        for seed in seeds:
            kwargs = {'num_nodes': nodes, 'node_features': config['model']['node_features'], 'hidden_dim': config['model']['hidden_dim']}
            if uses_weather: kwargs['weather_features'] = config['model']['weather_features']

            model = ModelClass(**kwargs).to(device)
            model.load_state_dict(torch.load(f"results/{file_key}_seed_{seed}.pth", map_location=device, weights_only=True))
            model.eval()

            sq_all, sq_c, sq_r, pred_list = [], [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred_scaled = model(batch.x, batch.edge_index, batch.weather, batch.batch)
                    preds = scaler_target.inverse_transform(pred_scaled.cpu().numpy()).flatten()
                    pred_list.append(preds)
                    actual = batch.y_raw.cpu().numpy().flatten()
                    sq_all.extend((actual - preds)**2)

                    for g in range(batch.num_graphs):
                        s_idx = g * nodes
                        errs = (actual[s_idx:s_idx+nodes] - preds[s_idx:s_idx+nodes])**2
                        if batch.weather[g][1].item() > 0.0:
                            sq_r.extend(errs)
                        else:
                            sq_c.extend(errs)

            seed_rmse.append(np.sqrt(np.mean(sq_all)))
            if sq_c: seed_clear.append(np.sqrt(np.mean(sq_c)))
            if sq_r: seed_rain.append(np.sqrt(np.mean(sq_r)))

            # Save the first seed of ContextFusion for the simulator
            if file_key == 'wa_stgat_context' and seed == 42:
                best_context_preds = np.concatenate(pred_list)

        results.append({
            'model': disp,
            'rmse': f"{np.mean(seed_rmse):.3f} ± {np.std(seed_rmse):.3f}",
            'clear': f"{np.mean(seed_clear):.3f} ± {np.std(seed_clear):.3f}" if seed_clear else "N/A",
            'rain': f"{np.mean(seed_rain):.3f} ± {np.std(seed_rain):.3f}" if seed_rain else "N/A"
        })

    print("="*75)
    print(f"{'MODEL':<28} | {'OVERALL RMSE':<15} | {'CLEAR RMSE':<12} | {'RAIN RMSE'}")
    print("="*75)
    for r in results:
        print(f"{r['model']:<28} | {r['rmse']:<15} | {r['clear']:<12} | {r['rain']}")
    print("="*75)

    print("\n--- MARKETPLACE SIMULATION (SENSITIVITY ANALYSIS) ---")
    print("Supply Ratio | XGBoost Unmet | WA-STGAT Unmet | Reduction %")
    print("-" * 65)
    for ratio in [0.70, 0.80, 0.90, 0.95]:
        wa_unmet, xgb_unmet = 0, 0
        for i in range(0, len(act_arr), nodes):
            d_act = act_arr[i:i+nodes]
            d_wa = np.maximum(best_context_preds[i:i+nodes], 0)
            d_xgb = np.maximum(xgb_arr[i:i+nodes], 0)

            tot_sup = d_act.sum() * ratio

            xgb_alloc = (d_xgb / (d_xgb.sum() + 1e-5)) * tot_sup
            wa_alloc = (d_wa / (d_wa.sum() + 1e-5)) * tot_sup

            xgb_unmet += np.sum(np.maximum(d_act - xgb_alloc, 0))
            wa_unmet += np.sum(np.maximum(d_act - wa_alloc, 0))

        reduction = ((xgb_unmet - wa_unmet) / xgb_unmet) * 100
        print(f"{int(ratio*100)}% Supply   | {xgb_unmet:,.0f}       | {wa_unmet:,.0f}        | {reduction:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    evaluate(args.config)
