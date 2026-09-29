#!/bin/bash


python main.py --train_data_type PoisonCIFAR10 --test_data_type CIFAR10 --config_path configs/cifar10 --num_of_workers 6 \
            --perturb_type classwise \
            --jpeg_defense \
            --use_generator \
            --filter_type low   \
            --cutoff 0.5 \
            --version resnet18 \
            --img_denoise 70 \
            --generator_filepath to/your/path