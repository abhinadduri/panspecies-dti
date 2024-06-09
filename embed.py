import argparse

from featurizers import ProtBertFeaturizer

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

    f = open(data_file)
    headers = next(f)
    assert headers.strip().split(',')[-1] == "Target Sequence"

    for line in f:
        prot_seq = line.strip().split(',')[-1]
        prot_list.append(prot_seq)


    protbert.write_to_disk(prot_list, file_path=out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args.data_file)
