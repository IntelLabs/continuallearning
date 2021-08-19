#!/bin/sh


python ../code_online/best_model/main_online_best_model.py \
-a resnet50 \
--lr 0.0125 \
--weight-decay 0.0001 \
--weight_old_data 1.0 \
--batch-size 256 \
--NOSubBatch 4 \
--workers 64 \
--gradient_steps_per_batch 1 \
--size_replay_buffer 40000 \
--epochs 90 \
--LR_adjust_intv 5 \
--cell_id "../data_preparation/release/cellID_yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250.npy" \
--root "/export/share/t1-datasets/yfcc100m_full_dataset_alt/images/" \
--data "../data_preparation/release/" \
--data_val "../data_preparation/release/yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250_valid_files_2004To2014_compact_val.csv" \
--dist-url 'tcp://127.0.0.1:23794' --dist-backend 'nccl' --world-size 1 --rank 0 --multiprocessing-distributed
