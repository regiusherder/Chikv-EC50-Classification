import argparse
import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm

def load_dataframes(file_paths):
    """Loads multiple CSVs into dataframes and extracts feature column names."""
    dataframes = [pd.read_csv(fp) for fp in file_paths]
    feature_columns = [list(df.columns) for df in dataframes]
    for cols in feature_columns:
        cols.remove('pIC50')
        cols.remove('Name')
    return dataframes, feature_columns

def load_models_and_scalers(model_paths, scaler_paths):
    """Loads trained models and scalers from pickle files."""
    models = [pickle.load(open(mp, 'rb')) for mp in model_paths]
    scalers = [pickle.load(open(sp, 'rb')) for sp in scaler_paths]
    return models, scalers

def process_screening_data(screening_path, feature_columns, scalers, models):
    """Processes screening data and generates predictions."""
    screening_df = pd.read_csv(screening_path).fillna(0)
    names = screening_df['Name']
    screening_df = screening_df[screening_df.isin([np.inf, -np.inf]).sum(axis=1) == 0]
    
    predictions = []
    for i in tqdm(range(5), desc="Predicting", unit="model"):
        scaled_features = scalers[i].transform(screening_df[feature_columns[i]])
        predictions.append(models[i].predict(scaled_features))
    
    screening_df['Classification'] = ['above_5' if sum(p == 'above_5' for p in pred) >= 3 else 'below_5' 
                                      for pred in zip(*predictions)]
    
    return screening_df[['Name', 'Classification']]

def main():
    parser = argparse.ArgumentParser(description="Predict molecular activity using trained models.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to input CSV file.")
    parser.add_argument('-o', '--output', type=str, default="predictions.csv", help="Path to output CSV file.")
    
    args = parser.parse_args()
    
    framework_paths = [f'training_data/data.csv'] * 5
    model_paths = [f'weights/models/frame_{i+1}.pkl' for i in range(5)]
    scaler_paths = [f'weights/scalers/frame_{i+1}.pkl' for i in range(5)]
    
    frameworks, feature_columns = load_dataframes(framework_paths)
    models, scalers = load_models_and_scalers(model_paths, scaler_paths)
    
    predictions_df = process_screening_data(args.input, feature_columns, scalers, models)
    predictions_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()