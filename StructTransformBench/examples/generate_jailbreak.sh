echo ""
# Generate jailbreak dataset if it doesn't exist
JAILBREAK_DATASET="jailbreak_dataset_full.json"
if [ ! -f "$JAILBREAK_DATASET" ]; then
    echo "🔥 Generating jailbreak attack dataset..."
    if python generate_jailbreak_dataset.py --output "$JAILBREAK_DATASET" --dataset-size 50 --methods PAIR WildteamAttack Jailbroken; then
        echo "✅ Jailbreak dataset generation completed"
    else
        echo "⚠️  Jailbreak dataset generation failed, continuing without pre-generated dataset..."
    fi
else
    echo "✅ Using existing jailbreak dataset: $JAILBREAK_DATASET"
fi
