#! /bin/bash

cd ../
python build_rawframes.py /vdata/dataset/Avenue_Dataset/training_videos/ /vdata/dataset/Avenue_Dataset/training_rawframes/ --level 1
echo "Raw frames RGB  Generated"
cd avenue/