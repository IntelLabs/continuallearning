#!/bin/sh

#python main_v3.py \
python ../code_offline/main.py \
-a resnet50 \
--lr 0.025 \
--workers 64 \
--batch-size 256 \
--adjust_lr 1 \
--num_passes 1 \
--use_aug 1 \
--use_val 1 \
--val_freq 1 \
--epochs 90 \
--used_data_rate_start 0.0 \
--used_data_rate_end 1.0 \
--evaluate \
--resume 'results_offline/resnet50_lr0.025_epochs90_numPasses1_useVal1_valFreq1_BS256/checkpoint.pth.tar' \
--cell_id "../data_preparation/release/cellID_yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250.npy" \
--root "../data_preparation/release/dataset/images/" \
--data "../data_preparation/release/" \
--data_val "../data_preparation/release/yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250_valid_files_2004To2014_compact_val.csv" \
--dist-url 'tcp://127.0.0.1:52176' --dist-backend 'nccl' --world-size 1 --rank 0 --multiprocessing-distributed

#val_data_rate = 0.5/0.01
# --resume '/mnt/beegfs/tier1/vcl-nfs-work/zcai/WorkSpace/continual_learning/training/code_github/Continual-Learning/results/resnet50_lr0.1_isOnline0_valDataRate0.001_adjustLr1_numPasses3_useAug1_useVal1_valFreq20/checkpoint.pth.tar' \
#/export/share/Datasets/yfcc100m_full_dataset/metadata_geolocation
