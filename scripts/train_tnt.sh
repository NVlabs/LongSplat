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

scene=("Barn" "Church" "Family" "Francis" "Horse" "Ignatius" "Museum")

for scene in "${scene[@]}"; do
    timestamp=$(date "+%Y-%m-%d_%H:%M:%S")
    python train.py --eval -s "./data/Tanks/$scene" -m "outputs/tanks/$scene/$timestamp" --port $port --mode tanks --disable_resize
    python render.py -m "outputs/tanks/$scene/$timestamp"
    python metrics.py -m "outputs/tanks/$scene/$timestamp"
done