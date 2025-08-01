import os
import sys
import argparse
import subprocess
import time
import requests
import json
import hashlib
from datetime import datetime

# OFFLINE_MODE_PATCH_APPLIED
# Hugging Face 오프라인 모드 설정
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/ceph_data/ycyoon/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.getcwd())
from easyjailbreak.attacker.WildteamAttack import WildteamAttack
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel
import torch


# First, prepare models and datasets.
TARGET_BASE_URL = "http://localhost:8001/v1"  # Target model (Llama 3.2 3B)
ATTACK_BASE_URL = "http://localhost:8002/v1"  # Attack model (Llama 3.1 70b R1)
EVAL_BASE_URL = "http://localhost:8003/v1"    # Eval model (HarmBench)
# FINAL_EVAL_BASE_URL removed - using HarmBench evaluator only
REF_EVAL_BASE_URL = "http://localhost:8004/v1"    # Refusal eval (WildGuard)

API_KEY_ATTACK = "EMPTY"  # No API key needed for local vLLM servers
API_KEY_EVAL = "EMPTY"    # No API key needed for local vLLM servers
API_KEY = os.getenv("API_KEY", "EMPTY")

# Foundation models configuration
FOUNDATION_MODELS = {
    "llama-3.2-1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",  # Updated to use actual model
        "base_url": TARGET_BASE_URL,
        "api_key": "EMPTY"
    },
    "llama-3.2-3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",  # Updated to use actual model
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
parser = argparse.ArgumentParser(description='Run Wildteam attack on different models')
parser.add_argument('--target-model', type=str, default='llama-3.2-1b',
                    choices=list(FOUNDATION_MODELS.keys()),
                    help='Target model to attack')
parser.add_argument('--dataset-size', type=int, default=None,
                    help='Number of samples to use from dataset')
parser.add_argument('--action', type=str, default='generate',
                    choices=['generate', 'print', 'attack'],
                    help='Action to perform')
parser.add_argument('--use-jailbreak-dataset', type=str, default=None,
                    help='Use pre-generated jailbreak dataset instead of running attack model')
parser.add_argument('--jailbreak-method', type=str, default='WildteamAttack', 
                    choices=['PAIR', 'WildteamAttack', 'Jailbroken'],
                    help='Which jailbreak method to use from dataset (default: WildteamAttack)')
args = parser.parse_args()

# Function to start target model server dynamically
def start_target_server(model_name, port=8001, gpu="0"):
    """Start a vLLM server for the target model"""
    print(f"🚀 Starting target model server: {model_name} on port {port}")
    
    # Check if server is already running
    try:
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            current_model = response.json()["data"][0]["id"]
            if current_model == model_name:
                print(f"✅ Server already running with {model_name}")
                return None
            else:
                print(f"🔄 Server running with different model ({current_model}), restarting...")
                # Kill existing server
                kill_cmd = f"lsof -ti:{port} | xargs kill -9"
                subprocess.run(kill_cmd, shell=True, stderr=subprocess.DEVNULL)
                time.sleep(3)
    except:
        pass
    
    # Start new server
    cmd = [
        "/home/ycyoon/anaconda3/envs/rgtnet/bin/python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "4096",
        "--block-size", "16",
        "--enforce-eager",
        "--disable-custom-all-reduce",
        "--distributed-executor-backend", "mp"
    ]
    
    # Set environment for the server
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "TORCH",
        "NCCL_TIMEOUT": "1800",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn"
    })
    
    # Start server in background
    process = subprocess.Popen(
        cmd, 
        stdout=open(f"target_server_{port}.log", "w"),
        stderr=subprocess.STDOUT,
        env=env
    )
    
    # Wait for server to be ready
    print("⏳ Waiting for target server to be ready...")
    for i in range(60):  # Wait up to 5 minutes
        try:
            response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ Target server ready on port {port}")
                return process
        except:
            pass
        time.sleep(5)
        if i % 6 == 0:  # Print every 30 seconds
            print(f"⏳ Still waiting... ({i*5}s)")
    
    print("❌ Target server failed to start within timeout")
    return None

def get_free_gpu():
    """Find a GPU with enough free memory"""
    import subprocess
    
    try:
        # Run nvidia-smi to get GPU memory info
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                idx, free_mem = line.split(',')
                gpu_info.append((int(idx), int(free_mem)))
            
            # Sort by free memory (descending)
            gpu_info.sort(key=lambda x: x[1], reverse=True)
            
            # Return GPU with most free memory (if > 10GB)
            for idx, free_mem in gpu_info:
                if free_mem > 10000:  # 10GB in MB
                    print(f"📊 Selected GPU {idx} with {free_mem}MB free memory")
                    return str(idx)
        
    except Exception as e:
        print(f"⚠️ Error checking GPU availability: {e}")
    
    # Default to GPU 1 if check fails
    return "1"

# Select target model based on argument
model_config = FOUNDATION_MODELS[args.target_model]

# Start target model server
free_gpu = get_free_gpu()
target_server_process = start_target_server(model_config["model_name"], gpu=free_gpu)
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

dataset = JailbreakDataset('HarmBench')
if args.dataset_size:
    dataset._dataset = dataset._dataset[:args.dataset_size]

# Create logs directory structure
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs/WildteamAttack_{args.target_model}_{timestamp}"
SAVE_PATH = f"{LOG_DIR}/results"

# Ensure to create the save path and pkl_files directory
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs("pkl_files", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

pruner_config = {
        "model_name": "alisawuffles/roberta-large-wanli",
        "pruner_type": "nli",
        "num_tokens": 512,
        "threshold": 0.9,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# Then instantiate the recipe.
attacker = WildteamAttack(attack_model=attack_model,
                          schema_type="TEXT",
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
pkl_file = f"pkl_files/Wild-JSON-{args.target_model}-results-v1.pkl"

# Result caching functions
def get_model_hash(model_config):
    """Generate a unique hash for model configuration"""
    if isinstance(model_config, str):
        config_str = model_config
    else:
        config_str = model_config.get("model_name", str(model_config))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def get_cached_result(cache_dir, model_hash, attack_type="WildteamAttack"):
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

def save_cached_result(cache_dir, model_hash, result_data, attack_type="WildteamAttack"):
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

def load_jailbreak_dataset(dataset_path, method='WildteamAttack'):
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

print(f"📁 Results will be saved to: {SAVE_PATH}")
print(f"📁 Pickle file: {pkl_file}")

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
                
                # Simple evaluation (can be enhanced with eval models)
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
    # Original WildteamAttack logic
    # Check for cached results first
    cached_result = get_cached_result(CACHE_DIR, model_hash, "WildteamAttack")

    if cached_result:
        print(f"🎯 Using cached WildteamAttack results for model hash {model_hash}")
        print(f"📅 Cache created: {cached_result['timestamp']}")
        print(f"✅ WildteamAttack completed using cached results!")
        
        # Create summary from cached result
        summary_file = f"{LOG_DIR}/WildteamAttack_summary_{timestamp}.json"
        if 'result' in cached_result and 'summary_data' in cached_result['result']:
            with open(summary_file, 'w') as f:
                json.dump(cached_result['result']['summary_data'], f, indent=2)
            print(f"📊 Summary restored to: {summary_file}")
    else:
        print("🚀 Running new WildteamAttack...")

    try:
        if args.action == 'generate':
            print("🔄 Generating attack prompts...")
            attacker.generate_attack_prompts(pkl_file)
            print("✅ Attack prompts generated successfully")
        elif args.action == 'print':
            print("📋 Printing pickle file contents...")
            attacker.print_pkl_file(pkl_file)
        elif args.action == 'attack':
            if not cached_result:
                print("🚀 Starting attack on target model...")
                results = attacker.attack_target_model(pkl_file)
                print("✅ Attack completed successfully")
                
                # Create a summary file
                summary_file = f"{LOG_DIR}/WildteamAttack_summary_{timestamp}.json"
                try:
                    summary_data = {
                        "target_model": args.target_model,
                        "dataset_size": args.dataset_size or len(dataset._dataset),
                        "timestamp": timestamp,
                        "attack_type": "WildteamAttack",
                        "schema_type": "JSON",
                        "save_path": SAVE_PATH,
                        "results_available": True
                    }
                    
                    with open(summary_file, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                    print(f"📊 Summary saved to: {summary_file}")
                    
                    # Cache the results
                    result_data = {
                        "target_model": args.target_model,
                        "summary_data": summary_data,
                        "save_path": SAVE_PATH,
                        "pkl_file": pkl_file,
                        "summary_file": os.path.abspath(summary_file),
                        "attack_completed": True
                    }
                    save_cached_result(CACHE_DIR, model_hash, result_data, "WildteamAttack")
                    
                except Exception as e:
                    print(f"⚠️ Could not create summary file: {e}")

    except Exception as e:
        if not cached_result:
            print(f"❌ WildteamAttack failed: {e}")
            import traceback
            traceback.print_exc()

    finally:
        # Clean up target server and GPU memory
        if 'target_server_process' in locals() and target_server_process:
            print("🛑 Stopping target model server...")
            target_server_process.terminate()
            try:
                target_server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                target_server_process.kill()
            
            # Also kill any remaining processes on port 8001
            kill_cmd = "lsof -ti:8001 | xargs kill -9"
            subprocess.run(kill_cmd, shell=True, stderr=subprocess.DEVNULL)
            
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleaned up")
