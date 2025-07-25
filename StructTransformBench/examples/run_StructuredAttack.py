import os
import pickle
import logging
import yaml

from easyjailbreak.attacker.SchemaAttackUpdated import StructuredAttack
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.
# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load models from config
def create_model(model_config):
    return OpenaiModel(
        model_name=model_config["model_name"],
        base_url=model_config["base_url"],
        api_keys=os.getenv(model_config["api_key_env"], "EMPTY"),
        generation_config=model_config["generation_config"]
    )

target_model = create_model(config["models"]["target"])
attack_model = create_model(config["models"]["attack"])
eval_model = create_model(config["models"]["eval"])
refusal_eval_model = create_model(config["models"]["refusal_eval"])

# Load dataset and structures
dataset = JailbreakDataset('HarmBench')
structures_dict = config["structures"]
SAVE_PATH = config["paths"]["save_path"]

# Load the attack config
attack_config = config.get('attack_settings', {})
combination_mode = attack_config.get('combination_mode', 1)  # Default to mode 1

# Ensure to create the save path
os.makedirs(SAVE_PATH, exist_ok=True)

failed_indices = range(len(dataset))

for structure_type in structures_dict.keys():
    for combined_flag in [False, True]:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)


        # Create directory for specific path
        STRUCTURE_SAVE_PATH = os.path.join(SAVE_PATH, (structure_type + "Combined") if combined_flag else structure_type)
        os.makedirs(STRUCTURE_SAVE_PATH, exist_ok=True)

        pkl_file_path = structures_dict[structure_type]
        print("=" * 50 + "STARTING STRUCTURE: " + STRUCTURE_SAVE_PATH + "=" * 50)
        attacker = StructuredAttack(attack_model=attack_model,
                                    schema_type=structure_type,
                                    combination_1=combined_flag if combination_mode == 1 else False,
                                    combination_2=combined_flag if combination_mode == 2 else False,
                                    target_model=target_model,
                                    eval_model=eval_model,
                                    refusal_eval_model=refusal_eval_model,
                                    jailbreak_datasets=dataset,
                                    max_queries=None,
                                    use_mutations=False,
                                    mutations_num=5,
                                    use_repeated=False,
                                    repeated_num=5,
                                    use_same_repeated=False,
                                    same_repeated_num=10,
                                    parallelize=True,
                                    num_threads=50,
                                    save_path= STRUCTURE_SAVE_PATH)

        pkl_save_path = f"{STRUCTURE_SAVE_PATH}/Responses.pkl"

        temp_dict = {}
        attack_dict = {}
        try:
            with open(pkl_file_path, 'rb') as file:
                temp_dict = pickle.load(file)
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()

        for i, (key, value) in enumerate(temp_dict.items()):
            if i in failed_indices:
                attack_dict[key] = temp_dict[key]



        attacker.attack(attack_dict, pkl_save_path)
        attacker.cleanup()
