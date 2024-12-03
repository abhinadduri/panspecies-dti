import pandas as pd
import argparse
from functools import partial
from multiprocessing import Pool, current_process
import os
import time
import numpy as np
import re

def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """
    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(' +', ' ', line).strip()
            splits = line.split(" ")
            
            if splits[0] == "ATOM":
                # If position < 1000
                if len(splits[4]) == 1:
                    pos = int(splits[5])
                
                # If position >= 1000, the blank will be removed, e.g. "A 999" -> "A1000"
                # So the length of splits[4] is not 1
                else:
                    pos = int(splits[4][1:])
                
                plddt = float(splits[-2])
                
                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)
    
    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts

def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    # assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    # check if the pdb file is empty
    assert os.path.getsize(path) > 0, f"PDB file is empty: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                plddts = extract_plddt(path)
                assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                
                # Mask regions with plddt < threshold
                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(struc_seq))
                np_seq[indices] = "#"
                struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I","--input-file", type=str, required=True, help='Structure file to extract SaProt sequence from')
    parser.add_argument('-C','--chain', type=str, default='A', help='Chain to extract from PDB file')
    parser.add_argument("-O","--output-csv", type=str, required=True, help='Filename to save the SaProt sequence to')
    parser.add_argument('--no-plddt-mask', action="store_false", help="do not use plddt mask for foldseek, use this flag if the pdb file does not have plddt scores")
    args = parser.parse_args()

    assert os.path.exists(args.input_file), f"PDB file not found: {args.input_file}"
    # check if foldseek is in the path
    assert os.system("which foldseek") == 0, "Foldseek not found in the path, please install foldseek (https://github.com/steineggerlab/foldseek) and add it to the path"
    seq_dict = get_struc_seq(foldseek="foldseek", path=args.input_file, chains=[args.chain], plddt_mask=args.no_plddt_mask)
    seq, struc_seq, combined_seq = seq_dict[args.chain]

    # open the output file and append a new row with the combined_sequence to the "Target Sequence" column and the pdb file name to the "uniprot_id" column
    if os.path.exists(args.output_csv):
        df = pd.read_csv(args.output_csv)
    else:
        df = pd.DataFrame(columns=["uniprot_id", "Target Sequence"])
    df = pd.concat([df,pd.DataFrame.from_dict({"uniprot_id": [os.path.basename(args.input_file)], "Target Sequence": [combined_seq]})], ignore_index=True)
    df.to_csv(args.output_csv)

if __name__ == "__main__":
    main()
