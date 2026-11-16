#!/bin/bash
# Full experiment matrix for the SSL camera+radar redesign.
# Edit the paths, then run stages one by one (each is a long GPU job).
set -euo pipefail

DATA=../nuscenes
DAV2_SRC=../Depth-Anything-V2                       # github.com/DepthAnything/Depth-Anything-V2
DAV2_W=../ckpts/depth_anything_v2_metric_vkitti_vits.pth
OFFICIAL_RGB=checkpoints/8x5_5e-4_rgb12_22:43:46     # sh get_rgb_model.sh
OFFICIAL_RAD=checkpoints/8x5_5e-4_rad25_18:55:34     # sh get_rad_model.sh
GPUS="[0,1,2,3]"
COMMON="--data_dir=$DATA --dav2_source=$DAV2_SRC --dav2_weights=$DAV2_W --device_ids=$GPUS --batch_size=8 --grad_acc=5 --max_iters=25000"

stage=${1:-help}
case $stage in
  pretrain)      # main model + ablations
    python train_ssl.py --exp_name=full $COMMON
    python train_ssl.py --exp_name=no_radar --use_radar=False $COMMON
    python train_ssl.py --exp_name=no_crossmodal --weights='{"sem_rad":0,"occ_cam":0,"occ_rad":0}' $COMMON
    python train_ssl.py --exp_name=no_moddrop --p_drop_camera=0 --p_drop_radar=0 $COMMON
    ;;
  probe)         # usage: ./ssl_runner.sh probe <ssl_ckpt_dir> <tag>
    CK=$2; TAG=$3
    for R in linear decoder; do
      for B in fused cam rad; do
        python probe_bev.py --model_type=ssl --init_dir=$CK --branch=$B --readout=$R \
          --data_dir=$DATA --out=probe_results/${TAG}_${B}_${R}.json
      done
      for M in empty wrong; do
        python probe_bev.py --model_type=ssl --init_dir=$CK --branch=fused --readout=$R \
          --radar_mode=$M --data_dir=$DATA --out=probe_results/${TAG}_fused_${R}_radar-${M}.json
      done
    done
    ;;
  baselines)     # identical probes on official Simple-BEV and ImageNet / random init
    for R in linear decoder; do
      python probe_bev.py --model_type=segnet --init_dir=$OFFICIAL_RGB --use_radar=False --readout=$R \
        --data_dir=$DATA --out=probe_results/simplebev_rgb_${R}.json
      python probe_bev.py --model_type=segnet --init_dir=$OFFICIAL_RAD --readout=$R \
        --data_dir=$DATA --out=probe_results/simplebev_rad_${R}.json
      python probe_bev.py --model_type=ssl --readout=$R \
        --data_dir=$DATA --out=probe_results/imagenet_ssl-arch_${R}.json
    done
    python probe_bev.py --model_type=segnet --init_dir=$OFFICIAL_RAD --readout=model --radar_mode=empty \
      --data_dir=$DATA --out=probe_results/simplebev_rad_model_radar-empty.json
    ;;
  lowlabel)      # usage: ./ssl_runner.sh lowlabel <ssl_ckpt_dir>
    CK=$2
    for F in 0.01 0.1 1.0; do
      python train_nuscenes.py --exp_name=ft_ssl_${F} --model_type=ssl --init_ckpt=$CK --use_radar=True \
        --label_frac=$F --nsweeps=5 --data_dir=$DATA --device_ids=$GPUS --max_iters=25000
      python train_nuscenes.py --exp_name=ft_imagenet_${F} --model_type=ssl --use_radar=True \
        --label_frac=$F --nsweeps=5 --data_dir=$DATA --device_ids=$GPUS --max_iters=25000
    done
    # then: python probe_bev.py --model_type=ssl --init_dir=<ft ckpt> --readout=model ...
    ;;
  *) echo "usage: $0 pretrain | probe <ckpt> <tag> | baselines | lowlabel <ckpt>";;
esac
