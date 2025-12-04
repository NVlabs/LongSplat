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

scene=("forest1" "forest2" "forest3" "garden1" "garden2" "garden3" "indoor" "playground" "university1" "university2" "university3" "university4")

for scene in "${scene[@]}"; do
    timestamp=$(date "+%Y-%m-%d_%H:%M:%S")
    python train.py --eval -s ./data/hike/$scene -m outputs/hike/$scene/"$timestamp" -r 4 --port $port --mode hike
    python render.py -m outputs/hike/$scene/"$timestamp"
    python metrics.py -m outputs/hike/$scene/"$timestamp"
done