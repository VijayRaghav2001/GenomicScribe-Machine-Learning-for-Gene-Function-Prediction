import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate gene function prediction model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    
    args = parser.parse_args()
    
    # Load model and data
    model = joblib.load(args.model)
    test_df = pd.read_csv(args.data)
    
    # Prepare features and labels
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels=['Non-Kinase', 'Kinase'])
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Example: save results to CSV
    results_df = pd.DataFrame(results['classification_report']).transpose()
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/evaluation_metrics.csv')
    
    print("Evaluation complete. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
