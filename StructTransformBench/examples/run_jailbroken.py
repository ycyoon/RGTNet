import os, sys
import argparse
import json
import hashlib
from datetime import datetime
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.Jailbroken_wei_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
sys.path.append(os.getcwd())

# Update model configurations and API keys
TARGET_BASE_URL = "http://localhost:8001/v1"  # Target model (Llama 3.2 3B)
ATTACK_BASE_URL = "http://localhost:8002/v1"  # Attack model (DeepSeek R1)
EVAL_BASE_URL = "http://localhost:8003/v1"    # Eval model (HarmBench)
# FINAL_EVAL_BASE_URL removed - using HarmBench evaluator only
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
parser = argparse.ArgumentParser(description='Run Jailbroken attack on different models')
parser.add_argument('--target-model', type=str, default='llama-3.2-1b',
                    choices=list(FOUNDATION_MODELS.keys()),
                    help='Target model to attack')
parser.add_argument('--dataset-size', type=int, default=None,
                    help='Number of samples to use from dataset')
parser.add_argument('--use-jailbreak-dataset', type=str, default=None,
                    help='Use pre-generated jailbreak dataset instead of running attack model')
parser.add_argument('--jailbreak-method', type=str, default='Jailbroken', 
                    choices=['PAIR', 'WildteamAttack', 'Jailbroken'],
                    help='Which jailbreak method to use from dataset (default: Jailbroken)')
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

attack_model = OpenaiModel(model_name="meta-llama/Llama-3.1-70B-Instruct",
                           base_url=ATTACK_BASE_URL,
                           api_keys=API_KEY_ATTACK,
                           generation_config={
                               "temperature": 1,
                               "top_p": 0.9,
                               "max_tokens": 1024
                           })

eval_model = OpenaiModel(model_name='cais/HarmBench-Llama-2-13b-cls',
                         base_url=EVAL_BASE_URL,
                         api_keys=API_KEY_EVAL,
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
if args.dataset_size:
    dataset._dataset = dataset._dataset[:args.dataset_size]
else:
    dataset._dataset = dataset._dataset[:]

# Result caching functions
def get_model_hash(model_config):
    """Generate a unique hash for model configuration"""
    if isinstance(model_config, str):
        config_str = model_config
    else:
        config_str = model_config.get("model_name", str(model_config))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def get_cached_result(cache_dir, model_hash, attack_type="Jailbroken"):
    """Check if cached result exists and return it"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{attack_type}_{model_hash}_cache.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            print(f"✅ Found cached {attack_type} result for model hash {model_hash}")
            return cached_data
        except Exception as e:
            print(f"⚠️ Error reading cache file: {e}")
    return None

def save_cached_result(cache_dir, model_hash, result_data, attack_type="Jailbroken"):
    """Save result to cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{attack_type}_{model_hash}_cache.json")
    
    try:
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "model_hash": model_hash,
            "attack_type": attack_type,
            "result": result_data
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"💾 Cached {attack_type} result for model hash {model_hash}")
        return True
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")
        return False

def load_jailbreak_dataset(dataset_path, method='Jailbroken'):
    """Load pre-generated jailbreak dataset"""
    try:
        import pickle
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        
        # Extract jailbreak prompts for the specified method
        jailbreak_prompts = []
        original_prompts = []
        
        dataset_items = data.get('data', data) if isinstance(data, dict) else data
        
        for item in dataset_items:
            original_prompt = item.get('original_prompt', '')
            attack_prompts = item.get('attack_prompts', {})
            
            if method in attack_prompts:
                jailbreak_prompt = attack_prompts[method]
            else:
                # Fallback to original prompt if method not found
                jailbreak_prompt = original_prompt
                print(f"⚠️ Method {method} not found for item {item.get('id', '?')}, using original prompt")
            
            original_prompts.append(original_prompt)
            jailbreak_prompts.append(jailbreak_prompt)
        
        print(f"✅ Loaded {len(jailbreak_prompts)} jailbreak prompts using method: {method}")
        return jailbreak_prompts, original_prompts
        
    except Exception as e:
        print(f"❌ Error loading jailbreak dataset: {e}")
        return None, None

SAVE_PATH = f"jailbroken-{args.target_model}-results"
os.makedirs(SAVE_PATH, exist_ok=True)

# Setup caching
CACHE_DIR = "attack_results_cache"
model_hash = get_model_hash(model_config)

# Check if using pre-generated jailbreak dataset
if args.use_jailbreak_dataset:
    print(f"\n🎯 Using pre-generated jailbreak dataset: {args.use_jailbreak_dataset}")
    print(f"🔧 Jailbreak method: {args.jailbreak_method}")
    
    # Load jailbreak dataset
    jailbreak_prompts, original_prompts = load_jailbreak_dataset(args.use_jailbreak_dataset, args.jailbreak_method)
    
    if jailbreak_prompts is None:
        print("❌ Failed to load jailbreak dataset. Exiting.")
        sys.exit(1)
    
    # Limit dataset size if specified
    if args.dataset_size and args.dataset_size < len(jailbreak_prompts):
        jailbreak_prompts = jailbreak_prompts[:args.dataset_size]
        original_prompts = original_prompts[:args.dataset_size]
        print(f"📊 Limited to {args.dataset_size} prompts")
    
    print(f"\n🚀 Running direct evaluation with pre-generated jailbreak prompts...")
    print(f"📊 Total prompts to test: {len(jailbreak_prompts)}")
    
    try:
        # Create results list
        results = []
        
        # Test each jailbreak prompt on target model
        for i, (original, jailbreak) in enumerate(zip(original_prompts, jailbreak_prompts)):
            print(f"\n📝 Testing prompt {i+1}/{len(jailbreak_prompts)}")
            print(f"Original: {original[:100]}...")
            print(f"Jailbreak: {jailbreak[:100]}...")
            
            try:
                # Get target model response
                target_response = target_model.generate(jailbreak)
                if isinstance(target_response, list):
                    target_response = target_response[0] if target_response else ""
                
                print(f"Response: {target_response[:200]}...")
                
                # Simple evaluation
                eval_result = "Unknown"
                
                result_entry = {
                    "index": i,
                    "original_prompt": original,
                    "jailbreak_prompt": jailbreak,
                    "target_response": target_response,
                    "eval_result": eval_result,
                    "method": args.jailbreak_method
                }
                results.append(result_entry)
                
            except Exception as e:
                print(f"❌ Error testing prompt {i+1}: {e}")
                continue
        
        # Save results
        result_file = f'{args.target_model}_{args.jailbreak_method}_dataset_result.jsonl'
        with open(result_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"\n✅ Dataset-based evaluation completed!")
        print(f"📊 Results:")
        print(f"   • Total prompts: {len(results)}")
        print(f"   • Method used: {args.jailbreak_method}")
        print(f"   • Results saved to: {result_file}")
        
    except Exception as e:
        print(f"❌ Dataset-based evaluation failed: {e}")
        import traceback
        traceback.print_exc()

else:
    # Original Jailbroken attack logic
    # Check for cached results first
    cached_result = get_cached_result(CACHE_DIR, model_hash, "Jailbroken")

    if cached_result:
        print(f"🎯 Using cached Jailbroken attack results for model hash {model_hash}")
        print(f"📅 Cache created: {cached_result['timestamp']}")
        print(f"✅ Jailbroken attack completed using cached results!")
        
        # Copy cached result file if it exists
        if 'result' in cached_result and 'result_file' in cached_result['result']:
            cached_file = cached_result['result']['result_file']
            result_file = os.path.join(SAVE_PATH, f'HarmBench_jailbroken_{args.target_model}.jsonl')
            if os.path.exists(cached_file):
                import shutil
                shutil.copy2(cached_file, result_file)
                print(f"📋 Result file copied to: {result_file}")
    else:
        print("🚀 Running new Jailbroken attack...")
        
        attacker = Jailbroken(attack_model=attack_model,
                              target_model=target_model,
                              eval_model=eval_model,
                              refusal_eval_model=refusal_eval_model,
                              jailbreak_datasets=dataset,
                              save_path=SAVE_PATH,
                              use_cache=True)

        try:
            # Generate and cache attack prompts
            attacker.generate_attack_prompts()

            # Perform the attack (this will now use the cached prompts)
            attacker.attack()
            attacker.log()
            
            result_file = os.path.join(SAVE_PATH, f'HarmBench_jailbroken_{args.target_model}.jsonl')
            attacker.attack_results.save_to_jsonl(result_file)
            
            print(f"✅ Jailbroken attack completed successfully!")
            
            # Cache the results
            result_data = {
                "target_model": args.target_model,
                "result_file": os.path.abspath(result_file),
                "save_path": SAVE_PATH,
                "dataset_size": args.dataset_size or len(dataset._dataset),
                "attack_completed": True
            }
            save_cached_result(CACHE_DIR, model_hash, result_data, "Jailbroken")
            
        except Exception as e:
            print(f"❌ Jailbroken attack failed: {e}")
            import traceback
            traceback.print_exc()
