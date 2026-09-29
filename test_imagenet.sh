#!/bin/bash


python main.py --train_data_type PoisonImageNetMini --test_data_type ImageNetMini --config_path configs/imagenet-mini --num_of_workers 6  --train_batch_size 64 --eval_batch_size 128 \
            --perturb_type classwise \
            --jpeg_defense \
            --use_generator \
            --filter_type low   \
            --cutoff 0.5 \
            --version resnet18 \
            --img_denoise 70 \
            --generator_filepath to/your/path