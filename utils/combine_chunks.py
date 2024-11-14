#!/usr/bin/python3
import argparse
from glob import glob
import pandas as pd

from ultrafast.utils import TopK

def parse_args():
    parser = argparse.ArgumentParser(description="Combine chunks of similarity data")
    parser.add_argument("input", help="Input directory containing similarity data chunks")
    parser.add_argument("--in-suffix", default=".csv", help="Input file suffix")
    parser.add_argument("-O","--output-topk",required=True, help="Output file")
    parser.add_argument("--output-bad", default=None, help="Output file for bad files")
    parser.add_argument("-K", type=int, default=100, help="Number of top similarities to keep")
    return parser.parse_args()

def main():
    # read the files in the input directory that end in ".csv"
    args = parse_args()
    input_files = glob(args.input + f"/*{args.in_suffix}")
    print(len(input_files))
    topk = TopK(args.K)
    bad_files = []
    # iterate over the files and read the similarities
    for idx, input_file in enumerate(input_files):
        df = pd.read_csv(input_file)
        # if the dataframe does not have the column "CosineSimi", then add it to the list of bad files
        if idx % 500 == 0:
            print(idx)
        if "CosineSimi" not in df.columns:
            bad_files.append(input_file)
            continue
        topk.push_list(df["CosineSimi"], df["id"], df["SMILES"])
    # if there are bad files write them to the output-bad file
    if bad_files:
        if args.output_bad is not None:
            with open(args.output_bad, "w") as f:
                f.write("\n".join(bad_files))
        else:
            print("Bad files:")
            print("\n\t".join(bad_files))
    # write the topk similarities to the output-topk file
    with open(args.output_topk, "w") as f:
        for similarity, id, smi in topk.get():
            f.write(f"{similarity},{id},{smi}\n")

if __name__ == "__main__":
    main()
