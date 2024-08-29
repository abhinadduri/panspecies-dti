import argparse
import pandas as pd

from ultrafast.featurizers import ProtBertFeaturizer

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        default="./data/prots.tsv",
        help="Path to the file containing data in CSV file format, with the protein sequence appearing as the last column. The column name should be \"Target Sequence\"",
    )

def main(data_file: str):
    protbert = ProtBertFeaturizer()
    out_file = data_file + ".prot.h5"
    prot_list = []

    df = pd.read_csv(data_file)
    headers = df.columns
    assert "Target Sequence" in headers

    prot_list = df["Target Sequence"].to_list()

    protbert.write_to_disk(prot_list, file_path=out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args.data_file)
