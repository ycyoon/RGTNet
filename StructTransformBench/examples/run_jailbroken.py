import os, sys
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.Jailbroken_wei_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
sys.path.append(os.getcwd())

# Update model configurations and API keys
TARGET_BASE_URL = "http://localhost:1601/v1"
ATTACK_BASE_URL = "https://api.deepseek.com/"
EVAL_BASE_URL = "https://openrouter.ai/api/v1/"
FINAL_EVAL_BASE_URL = "http://localhost:2217/v1"
REF_EVAL_BASE_URL = "http://localhost:3315/v1"

API_KEY_ATTACK = os.getenv("DEEPSEEK_PLATFORM_API_KEY", "EMPTY")
API_KEY_EVAL = os.getenv("OPENROUTER_API_KEY", "EMPTY")
API_KEY = os.getenv("API_KEY", "EMPTY")


# target_model = OpenaiModel(model_name="meta-llama/Llama-3.2-3B-Instruct",
#                            api_keys="EMPTY",
#                            base_url=TARGET_BASE_URL,
#                            generation_config={
#                                "temperature": 0,
#                                "top_p": 1.0,
#                                "max_tokens": 1024
#                            })
target_model = OpenaiModel(model_name="meta-llama/llama-3.2-90b-vision-instruct",
                           api_keys=API_KEY_EVAL,
                           base_url=EVAL_BASE_URL,
                           generation_config={
                               "temperature": 0,
                               "top_p": 1.0,
                               "max_tokens": 1024
                           })

# target_model = OpenaiModel(model_name="meta-llama/Llama-3.2-3B-Instruct",
#                            api_keys="EMPTY",
#                            base_url=TARGET_BASE_URL,
#                            generation_config={
#                                "temperature": 0,
#                                "top_p": 1.0,
#                                "max_tokens": 1024
#                            })

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


dataset_name = 'HarmBench'
dataset = JailbreakDataset(dataset_name)
dataset._dataset = dataset._dataset[:]

SAVE_PATH = "jailbroken-llama3.2-90b-results"
os.makedirs(SAVE_PATH, exist_ok=True)

attacker = Jailbroken(attack_model=attack_model,
                      target_model=target_model,
                      eval_model=eval_model,
                      refusal_eval_model=refusal_eval_model,
                      jailbreak_datasets=dataset,
                      save_path=SAVE_PATH,
                      use_cache=True)

# Generate and cache attack prompts
attacker.generate_attack_prompts()

# Perform the attack (this will now use the cached prompts)
attacker.attack()
attacker.log()
attacker.attack_results.save_to_jsonl(os.path.join(SAVE_PATH, 'HarmBench_jailbroken.jsonl'))
