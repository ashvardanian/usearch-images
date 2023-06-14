#!/bin/bash

wget -O images.fbin https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images.fbin
wget -O images.usearch https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images.usearch

if [ $1 = "cc3m" ]
then
    wget -O images_part1.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images_part1.txt
    wget -O images_part2.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images_part2.txt
    wget -O images_part3.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images_part3.txt
    cat images_part*.txt >> images.txt && rm images_part*.txt
    wget -O texts.fbin https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/texts.fbin
    wget -O texts.usearch https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/texts.usearch
else
    wget -O images.txt https://huggingface.co/datasets/unum-cloud/gallery-$1/resolve/main/images.txt
fi