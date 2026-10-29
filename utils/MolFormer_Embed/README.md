# MoLFormer
[MoLFormer](https://github.com/IBM/molformer) is a chemical language model for generating representations of small molecules from their SMILES strings.

## Generating MoLFormer embeddings to train SPRINT
1. Follow the instructions in [MoLFormer's environment.md](https://github.com/IBM/molformer/blob/main/environment.md) to setup a new environment (distinct from the SPRINT environment) for running MoLFormer.

2. Download the pretrained checkpoint of MoLFormer from [IBM's Box](https://ibm.box.com/v/MoLFormer-data) into this directory.

3. Download all of the files and folders in [molformer/notebooks/pretrained_molformer](https://github.com/IBM/molformer/tree/main/notebooks/pretrained_molformer) into this folder.

4. Run `molformer_embed.py`, making sure to set the `--csv-dir` to the directory containing train.csv, val.csv, and test.csv. It will use the pretrained MoLFormer model to generate embeddings for all of the molecules in the csvs and generate a lmdb directory that can be used during normal SPRINT training. 

## Running a SPRINT model on MoLFormer embeddings

Once you have generated the embeddings using the above procedure all you need to do when training is specify: `--drug-featurizer MoLFormerFeaturizer` on the commandline and it will use the pre-computed embeddings in the lmdb for training.
