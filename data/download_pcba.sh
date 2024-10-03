#!/bin/bash

# lit_pcba contains the smiles files needed to perform evaluation
if [ -d "lit_pcba" ]; then
    echo "lit_pcba folder already exists. Skipping download and extraction."
else
    wget -O "full_data.tgz" http://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz
    mkdir lit_pcba
    tar -xzf full_data.tgz -C lit_pcba
    rm full_data.tgz
    echo "Download and extraction completed."
fi


# download sequence only and sequence + 3di tokens for pcba sequences
gdown 1_XtNSW5nYtMnA07ubhOAqBC4UnEkHrZ-
unzip pcba_sequence_3di_tokens.zip -d lit_pcba
rm pcba_sequence_3di_tokens.zip
