import os
from easyjailbreak.attacker.WildteamAttack import WildteamAttack
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel
import torch


# First, prepare models and datasets.
TARGET_BASE_URL = "http://localhost:1601/v1"
ATTACK_BASE_URL = "https://api.deepseek.com/"
EVAL_BASE_URL = "https://openrouter.ai/api/v1/"
FINAL_EVAL_BASE_URL = "http://localhost:2217/v1"
REF_EVAL_BASE_URL = "http://localhost:3315/v1"

API_KEY_ATTACK = os.getenv("DEEPSEEK_PLATFORM_API_KEY", "EMPTY")
API_KEY_EVAL = os.getenv("OPENROUTER_API_KEY", "EMPTY")
API_KEY = os.getenv("API_KEY", "EMPTY")


target_model = OpenaiModel(model_name="meta-llama/Llama-3.2-3B-Instruct",
                           api_keys="EMPTY",
                           base_url=TARGET_BASE_URL,
                           generation_config={
                               "temperature": 0,
                               "top_p": 1.0,
                               "max_tokens": 1024
                           })


attack_model = OpenaiModel(model_name="deepseek-chat",
                           base_url=ATTACK_BASE_URL,
                           api_keys=API_KEY_ATTACK,
                           generation_config={
                               "temperature": 1,
                               "top_p": 0.9,
                               "max_tokens": 1024
                           })

eval_model = OpenaiModel(model_name='cais/HarmBench-Llama-2-13b-cls',
                         base_url="http://localhost:2217/v1",
                         api_keys=API_KEY,
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

SAVE_PATH = "Wild-JSON-llama3.2-3b-results-v1"

# Ensure to create the save path
os.makedirs(SAVE_PATH, exist_ok=True)

pruner_config = {
        "model_name": "alisawuffles/roberta-large-wanli",
        "pruner_type": "nli",
        "num_tokens": 512,
        "threshold": 0.9,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# Then instantiate the recipe.
attacker = WildteamAttack(attack_model=attack_model,
                          schema_type="JSON",
                          target_model=target_model,
                          eval_model=eval_model,
                          refusal_model=refusal_eval_model,
                          jailbreak_datasets=dataset,
                          max_queries=None,
                          use_mutations=False,
                          mutations_num=3,
                          use_repeated=True,
                          repeated_num=10,
                          parallelize=True,
                          num_threads=200,
                          pruner_config=pruner_config,
                          save_path=SAVE_PATH)

# Finally, start jailbreaking.
attacker.generate_attack_prompts("pkl_files/Wild-JSON-llama3.2-3b-results-v1.pkl")
# attacker.print_pkl_file("pkl_files/Wild-JSON-llama3.2-3b-results-v1.pkl")
# attacker.attack_target_model("pkl_files/Wild-JSON-llama3.2-3b-results-v1.pkl")
