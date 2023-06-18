#!/bin/bash

mkdir -p datasets/$1
wget -c -O datasets/$1/images.fbin https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images.fbin
wget -c -O datasets/$1/images.usearch https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images.usearch

if [ $1 = "cc3m" ]
then
    wget -c -O datasets/$1/images_part1.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images_part1.txt
    wget -c -O datasets/$1/images_part2.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images_part2.txt
    wget -c -O datasets/$1/images_part3.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images_part3.txt
    cat datasets/$1/images_part1.txt datasets/$1/images_part2.txt datasets/$1/images_part3.txt > datasets/$1/images.txt && rm datasets/$1/images_part*.txt
    wget -c -O datasets/$1/texts.fbin https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/texts.fbin
    wget -c -O datasets/$1/texts.usearch https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/texts.usearch
else
    wget -c -O datasets/$1/images.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images.txt
fi