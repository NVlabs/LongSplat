# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

ulimit -n 4096
port=$(rand 10000 30000)
scenes=("apple/110_13051_23361" "bench/415_57112_110099" "hydrant/106_12648_23157" "skateboard/245_26182_52130" "teddybear/34_1403_4393")

for scene in "${scenes[@]}"; do
    timestamp=$(date "+%Y-%m-%d_%H:%M:%S")
    python train.py --eval -s ./data/co3d/$scene -m outputs/co3d/$scene/"$timestamp" --port $port --mode co3d
    python render.py -m outputs/co3d/$scene/"$timestamp"
    python metrics.py -m outputs/co3d/$scene/"$timestamp"
done