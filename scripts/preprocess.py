import os
import yaml
import argparse
import requests
import urllib.request
import pandas as pd
import numpy as np
import torch


def preprocess(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    print("--- 1. Fetching TLC Data ---")
    file_name = config['paths']['tlc_data']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']


    if not os.path.exists(file_name):
        print(f"Downloading NYC TLC Data...")
        urllib.request.urlretrieve("https://d37ci6vzurychx.cloudfront.net/trip-data/" + file_name, file_name)


    df_temp = pd.read_parquet(file_name, engine='pyarrow')
    df_temp['tpep_pickup_datetime'] = pd.to_datetime(df_temp['tpep_pickup_datetime'])
    df_temp['tpep_dropoff_datetime'] = pd.to_datetime(df_temp['tpep_dropoff_datetime'])
    df_temp['trip_duration_mins'] = (df_temp['tpep_dropoff_datetime'] - df_temp['tpep_pickup_datetime']).dt.total_seconds() / 60.0


    mask = (
        (df_temp['passenger_count'] > 0) & (df_temp['fare_amount'] > 0) & 
        (df_temp['trip_distance'] > 0) & (df_temp['trip_distance'] < 100) &             
        (df_temp['trip_duration_mins'] >= 1) & (df_temp['trip_duration_mins'] <= 180) &       
        (df_temp['tpep_pickup_datetime'] >= start_date) & 
        (df_temp['tpep_pickup_datetime'] < '2024-01-04') # Demo boundary
    )
    df_clean = df_temp[mask].copy()
    df_clean['time_bin'] = df_clean['tpep_pickup_datetime'].dt.floor('15min')


    print("\n--- 2. Building Spatio-Temporal Grid ---")
    demand_df = df_clean.groupby(['time_bin', 'PULocationID']).size().reset_index(name='demand')
    all_times = pd.date_range(start=f'{start_date} 00:00:00', end=f'{end_date} 23:45:00', freq='15min')
    all_zones = range(1, 264) 
    idx = pd.MultiIndex.from_product([all_times, all_zones], names=['time_bin', 'PULocationID'])
    demand_matrix = demand_df.set_index(['time_bin', 'PULocationID']).reindex(idx, fill_value=0).reset_index().sort_values(by=['time_bin', 'PULocationID'])


    print("\n--- 3. Fetching Historical Weather ---")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=40.7143&longitude=-74.006&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,precipitation&timezone=America%2FNew_York"
    )
    weather_data = requests.get(url).json()
    real_weather_df = pd.DataFrame({
        'time_bin': pd.to_datetime(weather_data['hourly']['time']),
        'temperature': weather_data['hourly']['temperature_2m'],
        'rain_intensity': weather_data['hourly']['precipitation']
    }).set_index('time_bin').resample('15min').ffill().reset_index()


    print("\n--- 4. Final Merge & Engineering Lags ---")
    demo_df = pd.merge(demand_matrix, real_weather_df, on='time_bin', how='left')
    demo_df['day_of_week'] = demo_df['time_bin'].dt.dayofweek
    demo_df['hour'] = demo_df['time_bin'].dt.hour
    demo_df['minute'] = demo_df['time_bin'].dt.minute


    for lag in range(1, 5):
        demo_df[f'demand_lag_{lag}'] = demo_df.groupby('PULocationID')['demand'].shift(lag)


    demo_df = demo_df.dropna().reset_index(drop=True)
    os.makedirs('data', exist_ok=True)
    
    out_tensor = config['paths']['processed_tensor']
    demo_df.to_parquet(out_tensor)
    print(f"Tensor saved: {out_tensor}")


    print("\n--- 5. Generating Adjacency Matrix ---")
    valid_trips = df_clean[(df_clean['PULocationID'] <= 263) & (df_clean['DOLocationID'] <= 263)].copy()
    transition_counts = valid_trips.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')


    strong_edges = transition_counts[transition_counts['trip_count'] > config['data']['threshold']].copy()
    source_nodes = strong_edges['PULocationID'].values - 1
    target_nodes = strong_edges['DOLocationID'].values - 1
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)


    out_graph = config['paths']['edge_index']
    torch.save(edge_index, out_graph)
    print(f"Topology saved: {out_graph}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    preprocess(args.config)

