#!/bin/bash -ex
for i in 1 2 3 4 5
do
  for DS in 'Principles'
  do
   python src/main.py --input ../data/Principles --preprocessed-input preprocessed_dataset/Principles --d 512 --property n
   python src/main.py --input ../data/Principles --preprocessed-input preprocessed_dataset/Principles --d 512 --property s 
  done
done
