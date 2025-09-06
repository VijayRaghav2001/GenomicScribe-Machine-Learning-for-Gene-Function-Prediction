import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse
import yaml
import joblib

from models import get_model, train_model, hyperparameter_tuning

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description='Train gene function prediction model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load features
    features_df = pd.read_csv(config['data']['features_path'])
    
    # Prepare features and labels
    X = features_df.drop('label', axis=1).values
    y = features_df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['training']['test_size'], 
        random_state=config['training']['random_state']
    )
    
    # Train model
    if config['training']['hyperparameter_tuning']:
        print("Performing hyperparameter tuning...")
        best_model, best_params = hyperparameter_tuning(
            config['model']['name'], X_train, y_train, 
            config['model']['param_grid'], 
            cv=config['training']['cv_folds']
        )
        print(f"Best parameters: {best_params}")
    else:
        print("Training with default parameters...")
        model = get_model(config['model']['name'], config['model'].get('params', {}))
        best_model = train_model(model, X_train, y_train)
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_model, config['model']['save_path'])
    print(f"Model saved to {config['model']['save_path']}")

if __name__ == "__main__":
    main()
