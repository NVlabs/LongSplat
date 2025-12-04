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

scenes=("grass" "hydrant" "lab" "pillar" "road" "sky" "stair")

for scene in "${scenes[@]}"; do
    timestamp=$(date "+%Y-%m-%d_%H:%M:%S")
    python train.py --eval -s ./data/free/$scene -m outputs/free/$scene/"$timestamp" -r 2 --port $port --mode free
    python render.py -m outputs/free/$scene/"$timestamp"
    python metrics.py -m outputs/free/$scene/"$timestamp"
done