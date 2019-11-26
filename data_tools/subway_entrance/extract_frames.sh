#! /bin/bash

cd ../
python build_rawframes.py /vdata/dataset/events/subway_entrance/train/train_vedio/ /vdata/dataset/events/subway_entrance/train/train_rawframes/ --level 1
echo "Raw frames RGB  Generated"
cd subway_entrance/

