import os
import sys


sys.path.append(os.getcwd())
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.
TARGET_BASE_URL = "http://localhost:1601/v1"
ATTACK_BASE_URL = "https://api.deepseek.com/"
EVAL_BASE_URL = "https://openrouter.ai/api/v1/"
FINAL_EVAL_BASE_URL = "http://localhost:2217/v1"
REF_EVAL_BASE_URL = "http://localhost:3315/v1"

API_KEY_ATTACK = os.getenv("DEEPSEEK_PLATFORM_API_KEY", "EMPTY")
API_KEY_EVAL = os.getenv("OPENROUTER_API_KEY", "EMPTY")
API_KEY = os.getenv("API_KEY", "EMPTY")

# example target model 
target_model = OpenaiModel(model_name="meta-llama/Llama-3.2-3B-Instruct", api_keys="EMPTY", base_url=TARGET_BASE_URL)

# 加载attack model

attack_model = OpenaiModel(model_name="deepseek-chat", api_keys=API_KEY_ATTACK, base_url=ATTACK_BASE_URL)

'''
Diff between eval model and final eval model:
eval model: used by PAIR to improve its attack iteratively
final eval model: used at the end to judge if the final prompt is successful or no
'''

eval_model = OpenaiModel(model_name="openai/gpt-4o-mini", api_keys=API_KEY_EVAL,
                         base_url=EVAL_BASE_URL,
                         generation_config={
                             "temperature": 0.0,
                             "max_tokens": 5,
                         }, debug=1)

final_eval_model = OpenaiModel(model_name='cais/HarmBench-Llama-2-13b-cls',
                               base_url=FINAL_EVAL_BASE_URL,
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
dataset._dataset = dataset._dataset

SAVE_PATH = "PAIR-llama3.2-3b-results"

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
attacker.attack(save_path='Llama-3.2-3B-final_deepseek-chat_gpt_4o_mini_HarmBench_result.jsonl')
