#!/bin/bash

# VSDS_vd contains the .smi and .json files needed to perform evaluation
if [ -d "VSDS_vd" ]; then
    echo "VSDS_vd folder already exists. Skipping download and extraction."
else
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11ORm1S7LAlYancgMAa0NOvW5wJV6fUxp' -O vsds_vd_data.tgz
    tar -xzf vsds_vd_data.tgz
    rm vsds_vd_data.tgz
    echo "Download and extraction completed."
fi
