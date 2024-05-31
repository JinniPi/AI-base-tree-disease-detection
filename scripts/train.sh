#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

save_dir=data/pest
mkdir -p ${save_dir}

python analyze_model.py \
  --input_image_size 224 \
  --num_classes 102 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt


python train_image_classification.py --dataset pest --gpu 0 --num_classes 102 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size 224 --epochs 480 --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment --shuffle_train_data\
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ZenNet/ZenNet-IP102.txt \
  --load_parameters_from ${save_dir}/pest_480epochs/best-params_rank0.pth \
  --evaluate_only \
  --batch_size_per_gpu 128 \
  --print_freq 1000 \
  --save_dir ${save_dir}/pest_480epochs
