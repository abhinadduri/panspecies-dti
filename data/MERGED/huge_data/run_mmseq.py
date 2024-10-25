# This script runs MMSeq2 on the LIT-PCBA dataset to only retain uniprot id's with sequence similarity <90% to LIT-PCBA

import json
import numpy as np
import subprocess
import os
import sys

def run_mmseqs2(query_file, target_file, output_file, tmp_dir, threshold):
    commands = [
        f"mmseqs createdb {query_file} queryDB",
        f"mmseqs createdb {target_file} targetDB",
        f"mmseqs search queryDB targetDB resultDB {tmp_dir} -s 7.5 --max-seqs 10000 --min-seq-id {threshold}",
        f"mmseqs convertalis queryDB targetDB resultDB {output_file}"
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)

def write_fasta(sequences, filename):
    with open(filename, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")

def main(threshold=0.9):
    # Load LIT-PCBA sequences
    target_sets = json.load(open('lit_pcba_sequence_dict.json'))
    lit_pcba_sequences = {}
    # for k in target_sets:
    #     for idx, seq in enumerate(target_sets[k]):
    #         lit_pcba_sequences[idx] = seq

    idx = 0
    for k in target_sets:
        for seq in target_sets[k]:
            lit_pcba_sequences[idx] = seq
            idx += 1
    print(f"Number of LIT-PCBA sequences: {len(lit_pcba_sequences)}")

    # Load training data sequences
    train_seqs = np.load('id_to_sequence.npy', allow_pickle=True).item()
    id_list = [k for k in train_seqs.keys() if not any(char.isdigit() for char in train_seqs[k])]
    train_seqs = {k: train_seqs[k] for k in id_list}
    print(f"Number of training sequences: {len(train_seqs)}")

    # Write sequences to FASTA files
    write_fasta(lit_pcba_sequences, 'lit_pcba.fasta')
    write_fasta(train_seqs, 'train_seqs.fasta')

    # Run MMseqs2
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    run_mmseqs2('train_seqs.fasta', 'lit_pcba.fasta', 'similar_sequences.tsv', tmp_dir, threshold)

    # Process results
    similar_ids = set()
    with open('similar_sequences.tsv', 'r') as f:
        for line in f:
            query_id = line.split('\t')[0]
            similar_ids.add(query_id)

    # Write excluded UniProt IDs to file
    with open(f'temp_uniprots_excluded_at_{int(threshold*100)}.txt', 'w') as f:
        for uniprot_id in similar_ids:
            f.write(f"{uniprot_id}\n")

    print(f"Number of UniProt IDs to exclude: {len(similar_ids)}")
    subprocess.run("rm targetDB*; rm queryDB*; rm resultDB*; rm -rf tmp", shell=True, check=True)

if __name__ == "__main__":
    main(threshold=float(sys.argv[1]))
