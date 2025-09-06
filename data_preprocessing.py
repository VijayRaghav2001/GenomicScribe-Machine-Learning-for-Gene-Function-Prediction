import pandas as pd
from Bio import SeqIO
import argparse
import os

def parse_genbank(file_path):
    """Parse GenBank file and extract CDS information"""
    records = list(SeqIO.parse(file_path, "genbank"))
    data = []
    
    for record in records:
        for feature in record.features:
            if feature.type == "CDS":
                gene_data = {
                    "locus_tag": feature.qualifiers.get("locus_tag", [""])[0],
                    "gene": feature.qualifiers.get("gene", [""])[0],
                    "protein_id": feature.qualifiers.get("protein_id", [""])[0],
                    "product": feature.qualifiers.get("product", [""])[0],
                    "sequence": str(feature.extract(record.seq)),
                    "start": int(feature.location.start),
                    "end": int(feature.location.end),
                    "strand": feature.location.strand,
                    "go_terms": [x for x in feature.qualifiers.get("db_xref", []) if x.startswith("GO:")]
                }
                data.append(gene_data)
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Preprocess genomic data')
    parser.add_argument('--input', type=str, required=True, help='Input directory with raw data')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Process all GenBank files in the input directory
    all_data = []
    for file in os.listdir(args.input):
        if file.endswith('.gb') or file.endswith('.gbk'):
            df = parse_genbank(os.path.join(args.input, file))
            all_data.append(df)
    
    # Combine all data and save
    combined_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(args.output, exist_ok=True)
    combined_df.to_csv(os.path.join(args.output, 'processed_genes.csv'), index=False)
    print(f"Processed {len(combined_df)} genes")

if __name__ == "__main__":
    main()
