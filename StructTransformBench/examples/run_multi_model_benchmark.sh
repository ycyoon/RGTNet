#TARGET_MODEL="/home/ycyoon/work/RGTNet/models/rgtnet_final_model_20250801"
TARGET_MODEL="/home/ycyoon/work/RGTNet/models/merged_epoch_0"

python multi_model_benchmark.py --trained-model rgtnet $TARGET_MODEL --use-local --num_prompts -1
