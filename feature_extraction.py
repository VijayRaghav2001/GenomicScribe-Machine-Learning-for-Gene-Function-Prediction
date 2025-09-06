import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product
import argparse
import os

def calculate_gc_content(sequence):
    """Calculate GC content of a DNA sequence"""
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0

def calculate_kmer_frequencies(sequences, k=3):
    """Calculate k-mer frequencies for sequences"""
    kmers = [''.join(kmer) for kmer in product('ACGT', repeat=k)]
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), vocabulary=kmers)
    kmer_counts = vectorizer.fit_transform(sequences)
    return kmer_counts.toarray(), vectorizer.get_feature_names_out()

def extract_features(df):
    """Extract various features from gene sequences"""
    features = {}
    
    # Basic sequence features
    features['length'] = df['sequence'].apply(len)
    features['gc_content'] = df['sequence'].apply(calculate_gc_content)
    
    # K-mer frequencies
    kmer_counts, kmer_names = calculate_kmer_frequencies(df['sequence'], k=3)
    for i, name in enumerate(kmer_names):
        features[f'kmer_{name}'] = kmer_counts[:, i]
    
    # Additional features can be added here
    # ...
    
    return pd.DataFrame(features)

def main():
    parser = argparse.ArgumentParser(description='Extract features from gene sequences')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with processed genes')
    parser.add_argument('--output', type=str, required=True, help='Output file for features')
    
    args = parser.parse_args()
    
    # Load processed data
    df = pd.read_csv(args.input)
    
    # Extract features
    features_df = extract_features(df)
    
    # Add target labels (simplified for example)
    # In a real scenario, you would parse GO terms properly
    features_df['label'] = df['product'].apply(lambda x: 1 if 'kinase' in str(x).lower() else 0)
    
    # Save features
    features_df.to_csv(args.output, index=False)
    print(f"Extracted {features_df.shape[1]} features for {features_df.shape[0]} genes")

if __name__ == "__main__":
    main()
