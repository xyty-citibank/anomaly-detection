#! /bin/bash

cd ../
python build_rawframes.py /vdata/dataset/Avenue_Dataset/testing_videos/ /vdata/dataset/Avenue_Dataset/testing_rawframes/ --level 1
echo "Raw frames RGB  Generated"
cd avenue/