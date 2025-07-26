import os
import sys
import argparse

# OFFLINE_MODE_PATCH_APPLIED
# Hugging Face 오프라인 모드 설정
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/ceph_data/ycyoon/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.getcwd())
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.
TARGET_BASE_URL = "http://localhost:8001/v1"  # Target model (Llama 3.2 3B)
ATTACK_BASE_URL = "http://localhost:8002/v1"  # Attack model (DeepSeek R1)
EVAL_BASE_URL = "http://localhost:8003/v1"    # Eval model (HarmBench)
FINAL_EVAL_BASE_URL = "http://localhost:8006/v1"  # Final eval (Llama 4 Scout)
REF_EVAL_BASE_URL = "http://localhost:8004/v1"    # Refusal eval (WildGuard)

API_KEY_ATTACK = "EMPTY"  # No API key needed for local vLLM servers
API_KEY_EVAL = "EMPTY"    # No API key needed for local vLLM servers
API_KEY = os.getenv("API_KEY", "EMPTY")

# Foundation models configuration
FOUNDATION_MODELS = {
    "llama-3.2-1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "llama-3.2-3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "llama-3.1-8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "qwen-2.5-7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "qwen-2.5-14b": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "gemma-2-9b": {
        "model_name": "google/gemma-2-9b-it",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "phi-3.5-mini": {
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "vicuna-7b": {
        "model_name": "lmsys/vicuna-7b-v1.5",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "falcon-7b": {
        "model_name": "tiiuae/falcon-7b-instruct",
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    }
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run PAIR attack on different models')
parser.add_argument('--target-model', type=str, default='llama-3.2-1b',
                    choices=list(FOUNDATION_MODELS.keys()),
                    help='Target model to attack')
parser.add_argument('--dataset-size', type=int, default=None,
                    help='Number of samples to use from dataset')
args = parser.parse_args()

# Select target model based on argument
model_config = FOUNDATION_MODELS[args.target_model]
target_model = OpenaiModel(
    model_name=model_config["model_name"],
    api_keys=model_config["api_key"],
    base_url=model_config["base_url"],
    generation_config={
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 1024
    }
)

# example target model 
# target_model = OpenaiModel(model_name="meta-llama/Llama-3.2-3B-Instruct", api_keys="EMPTY", base_url=TARGET_BASE_URL)

# 加载attack model - Use DeepSeek model for better attack performance
attack_model = OpenaiModel(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 
                          api_keys=API_KEY_ATTACK, 
                          base_url=ATTACK_BASE_URL,
                          generation_config={
                              "temperature": 1.0,
                              "max_tokens": 500
                          })

'''
Diff between eval model and final eval model:
eval model: used by PAIR to improve its attack iteratively
final eval model: used at the end to judge if the final prompt is successful or no
'''

eval_model = OpenaiModel(model_name="cais/HarmBench-Llama-2-13b-cls", 
                         api_keys=API_KEY_EVAL,
                         base_url=EVAL_BASE_URL,
                         generation_config={
                             "temperature": 0.0,
                             "max_tokens": 5,
                         }, debug=1)

final_eval_model = OpenaiModel(model_name='meta-llama/Meta-Llama-3-70B-Instruct',
                               base_url=FINAL_EVAL_BASE_URL,
                               api_keys="EMPTY",
                               generation_config={
                                   "temperature": 0.0,
                                "max_tokens": 1
                                })

refusal_eval_model = OpenaiModel(model_name='allenai/wildguard',
                                 base_url=REF_EVAL_BASE_URL,
                                 api_keys="EMPTY",
                                 generation_config={
                                   "temperature": 0.0,
                                    "max_tokens": 20
                                })

dataset = JailbreakDataset('HarmBench')
dataset._dataset = dataset._dataset
if args.dataset_size:
    dataset._dataset = dataset._dataset[:args.dataset_size]

SAVE_PATH = f"PAIR-{args.target_model}-results"

# Ensure to create the save path
os.makedirs(SAVE_PATH, exist_ok=True)

# Then instantiate the recipe.
attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                final_eval_model=final_eval_model,
                refusal_eval_model=refusal_eval_model,
                eval_model=eval_model,
                n_streams=5,
                n_iterations=5,
                jailbreak_datasets=dataset,
                save_path=SAVE_PATH)

# Finally, start jailbreaking.
attacker.attack(save_path=f'{args.target_model}_deepseek-lamma4_scout17B_HarmBench_result.jsonl')
