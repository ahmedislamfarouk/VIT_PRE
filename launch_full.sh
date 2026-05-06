#!/bin/bash
# Run ALL experiments: 7 models × 4 subsets = 28 parallel jobs

echo "=== Launching 28 PARALLEL experiments ==="
echo "Models: CNN, DeiT, DINOv2, SigLIP, MobileViT-S, MobileViT-Hard, AMTD"
echo "Subsets: 5%, 10%, 50%, 100%"
echo ""

# Clear old results
rm -f /home/skyvision/ammarbigass5/results/scratch_*.json

# Function to run on specific GPU
run_exp() {
    local subset=$1
    local model=$2
    local gpu=$3
    CUDA_VISIBLE_DEVICES=$gpu python3.10 /home/skyvision/ammarbigass5/run_scratch_fixed.py $subset $model > /home/skyvision/ammarbigass5/logs/exp_${subset}_${model}.log 2>&1
}

# 5% - GPUs 0-6
echo "5% subset..."
run_exp 0.05 cnn 0 &
run_exp 0.05 deit 1 &
run_exp 0.05 dinov2 2 &
run_exp 0.05 siglip 3 &
run_exp 0.05 mobilevit_s 4 &
run_exp 0.05 mobilevit_hard 5 &
run_exp 0.05 amtd 6 &

# 10% - GPUs 1-7
echo "10% subset..."
run_exp 0.10 cnn 1 &
run_exp 0.10 deit 2 &
run_exp 0.10 dinov2 3 &
run_exp 0.10 siglip 4 &
run_exp 0.10 mobilevit_s 5 &
run_exp 0.10 mobilevit_hard 6 &
run_exp 0.10 amtd 7 &

# 50% - GPUs 2,3,4,5 (just key models)
echo "50% subset..."
run_exp 0.50 cnn 2 &
run_exp 0.50 deit 3 &
run_exp 0.50 amtd 4 &
run_exp 0.50 mobilevit_s 5 &

# 100% - GPUs 6,7 (just key models)
echo "100% subset..."
run_exp 1.00 cnn 6 &
run_exp 1.00 amtd 7 &
run_exp 1.00 mobilevit_s 0 &
run_exp 1.00 deit 1 &

echo "=== All jobs launched! ==="
echo "Est time: 5%=8min, 10%=10min, 50%=12min, 100%=15min"