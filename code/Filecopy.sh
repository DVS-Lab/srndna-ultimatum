#!/bin/bash

FILES="/data/projects/srndna-ultimatum/derivatives/singletrial/sub-*/sub-*_run-0*_mask-roi-*.txt"

for sub in $FILES
do
   #echo "This is the file: $sub"
   cp "$sub" /data/projects/srndna-ultimatum/temp
done 
