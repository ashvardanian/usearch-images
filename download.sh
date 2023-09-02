#!/bin/bash

mkdir -p data/$1
wget -c -O data/$1/images.fbin https://huggingface.co/datasets/unum-cloud/ann-$1/resolve/main/images.fbin
wget -c -O data/$1/images.uform-vl-multilingual-v2.fbin https://huggingface.co/datasets/unum-cloud/ann-$1/resolve/main/images.uform-vl-multilingual-v2.fbin
wget -c -O data/$1/images.vit-bigg-14-laion2b_s39b_b160k.fbin https://huggingface.co/datasets/unum-cloud/ann-$1/resolve/main/images.vit-bigg-14-laion2b_s39b_b160k.fbin

if [ $1 = "cc-3m" ]
then
    wget -c -O data/$1/images_part1.txt https://huggingface.co/datasets/unum-cloud/ann-$1/resolve/main/images_part1.txt
    wget -c -O data/$1/images_part2.txt https://huggingface.co/datasets/unum-cloud/ann-$1/resolve/main/images_part2.txt
    wget -c -O data/$1/images_part3.txt https://huggingface.co/datasets/unum-cloud/ann-$1/resolve/main/images_part3.txt
    cat data/$1/images_part1.txt data/$1/images_part2.txt data/$1/images_part3.txt > data/$1/images.txt && rm data/$1/images_part*.txt
else
    wget -c -O data/$1/images.txt https://huggingface.co/datasets/unum-cloud/ann-$1/resolve/main/images.txt
fi