#!/bin/bash
#Training and testing on scale factor 3.
# Train on DIV2K datasets.
python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 1 --rgb_range 1 --chunk_size 128 --n_hashes 3 --save_models --lr 1e-4 --decay 300-600-900-1200 --epochs 1500 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 256 --reduction 4 --res_scale 0.1 --batch_size 16 --model DLSN --scale 3 --patch_size 144 --save DLSN_x3 --data_train DIV2K
# Test on Set5, Set14, B100, Urban100, Manga109 datasets.
python main.py --dir_data ../../ --data_test Set5+Set14+B100+Urban100+Manga109 --n_GPUs 1 --rgb_range 1 --save_models --save_results --n_resgroups 10 --n_resblocks 4 --n_feats 256 --res_scale 0.1 --model DLSN --pre_train ../experiment/DLSN_x3/model/DLSN_x3.pt --save Temp --data_range 1-800/1-5 --scale 3 --test_only --reduction 4 --chunk_size 128 --n_hashes 3 --chop
