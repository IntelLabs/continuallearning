#!/bin/sh

python ../code_online/no_PoLRS/main_online.py \
-a resnet50 \
--lr 0.05 \
--weight-decay 0.0001 \
--batch-size 256 \
--NOSubBatch 1 \
--workers 32 \
--size_replay_buffer 40000 \
--val_freq 1000 \
--epochs 90 \
--adjust_lr 1 \
--SaveInter 1 \
--use_ADRep 1 \
--cell_id "../data_preparation/release/cellID_yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250.npy" \
--root "/export/share/t1-datasets/yfcc100m_full_dataset_alt/images/" \
--data "../data_preparation/release/" \
--data_val "../data_preparation/release/yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250_valid_files_2004To2014_compact_val.csv" \
--dist-url 'tcp://127.0.0.1:11805' --dist-backend 'nccl' --world-size 1 --rank 0 --multiprocessing-distributed
