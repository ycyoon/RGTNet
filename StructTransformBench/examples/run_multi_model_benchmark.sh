TARGET_MODEL="/home/ycyoon/work/RGTNet/models/rgtnet_llama-3.2-3b-instruct_20250802_2219/lora_adapters_epoch_1"

python multi_model_benchmark.py --trained-model rgtnet $TARGET_MODEL --use-local --num_prompts -1