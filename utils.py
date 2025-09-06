import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

def plot_precision_recall_curve(y_true, y_scores):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/precision_recall_curve.png')
    plt.close()
    
    return pr_auc

def load_go_terms(go_file_path):
    """Load Gene Ontology terms from file"""
    go_terms = {}
    with open(go_file_path, 'r') as file:
        for line in file:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                go_id, name = parts[0], parts[1]
                go_terms[go_id] = name
    return go_terms
