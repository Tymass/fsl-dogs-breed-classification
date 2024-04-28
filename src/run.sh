#!/bin/bash

URL_DATASET="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
URL_LISTS="http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
REQUIREMENTS_FILE="requirements.txt"

FILENAME_DATASET="dataset.tar"
FILENAME_LISTS="lists.tar"

curl -o "$FILENAME_LISTS" "$URL_LISTS"
curl -o "$FILENAME_DATASET" "$URL_DATASET"

rm -rf lists
rm -rf dataset
rm -rf models
mkdir dataset
mkdir dataset/lists
mkdir models
mkdir train_val_test

tar -xvf "$FILENAME_LISTS" -C "$PWD/dataset/lists" 
rm "$FILENAME_LISTS"
tar -xvf "$FILENAME_DATASET" -C "$PWD/dataset"
rm "$FILENAME_DATASET"

pip install -r "$REQUIREMENTS_FILE"

python3 dataset_setup.py

#rm -rf dataset

