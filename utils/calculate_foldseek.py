import pandas as pd
import argparse
from functools import partial
from multiprocessing import Pool, current_process
import os
import time
import numpy as np
import re
from ultrafast.utils import get_struc_seq

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


def calculate_combined_seq(uniprot_id, rowname="", plddt_mask=True):
    identifier, pdb_path = find_pdb_file(uniprot_id, rowname)
    if 'na' in pdb_path:
        print(uniprot_id, rowname)
    process_id = current_process().pid
    parsed_seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"], process_id=process_id, plddt_mask=plddt_mask)
    if type(parsed_seqs) == dict and 'A' in parsed_seqs:
        parsed_seqs = parsed_seqs['A']
    return identifier, parsed_seqs[2] if len(parsed_seqs) == 3 else ""

# given a df with uniprot column, find the pdb file for that uniprot ID. If uniprot ID is "NA" then look in the colabfold folder for the file with the name DataFrame file_row + ".pdb".
def find_pdb_file(uniprot_id, rowname):
    isna=False
    try:
        isna = np.isna(uniprot_id)
    except:
        pass

    if uniprot_id == "NA" or uniprot_id == "nan":
        pdb_file = "colabfold/" + str(rowname) + ".pdb"
        identifier = rowname
    elif "PDB" in uniprot_id:
        pdb_id = uniprot_id.split('_')[-1]
        pdb_file = f"refined_set_input/{pdb_id}/{pdb_id}_PROT.pdb"
        identifier = uniprot_id
    else:
        pdb_file = "AFDB/" + str(uniprot_id.replace('[','_').replace(']','_')) + ".pdb"
        identifier = uniprot_id
    return identifier, pdb_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_csv", type=str, required=True)
    parser.add_argument("-o","--output_csv", type=str, required=True)
    parser.add_argument('--no_plddt_mask', action="store_false", help="do not use plddt mask for foldseek")
    parser.add_argument("-p",'--processes', type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df['uniprot_id'] = df['uniprot_id'].astype(str)
    # get the unique uniprot IDs
    uniprot_ids = df["uniprot_id"].unique()
    uniprot_ids = uniprot_ids[uniprot_ids != 'nan']
    # find the row indices where the uniprot ID is "NA"
    if 'BIOSNAP' not in args.input_csv:
        df['rowname'] = '_'.join(args.input_csv.replace('.csv','_').split("/")[-2:]) + df.index.astype(str)
    else:
        df['rowname'] = '_'.join(['BIOSNAP',args.input_csv.split("/")[-1]]) + df.index.astype(str)
    na_rownames = df[df["uniprot_id"] == "nan"]["rowname"]
    # create a list of tuples with the uniprot ID and the row index
    foldseek_calc_list = list((uid, "") for uid in uniprot_ids)  + list(("nan", narow) for narow in na_rownames)

    # for each uniprot ID, find the pdb file and calculate the combined sequence
    # create a pool of workers to parallelize the process and save to a dictionary: {uniprot_id: combined_seq} or {file_row: combined_seq}
    calc_combined_seq = partial(calculate_combined_seq, plddt_mask=args.no_plddt_mask)
    with Pool(processes=args.processes) as p:
        results = p.starmap(calc_combined_seq, foldseek_calc_list)
    identifiers, combined_seq = zip(*results)
    combined_seq_dict = dict(zip(identifiers, combined_seq))
    # update the dataframe with the combined sequences
    def update_combined_seq(row):
        if row["uniprot_id"] == "nan":
            return combined_seq_dict[row["rowname"]]
        else:
            return combined_seq_dict[row["uniprot_id"]]
    df["Target Sequence"] = df.apply(update_combined_seq, axis=1)

    df.drop(columns=["rowname"], inplace=True)
    df.to_csv(args.output_csv)

if __name__ == "__main__":
    main()
