import os
import sys
import argparse
import subprocess
import time
import requests

# OFFLINE_MODE_PATCH_APPLIED
# Hugging Face 오프라인 모드 설정
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/ceph_data/ycyoon/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.getcwd())
sys.path.append('/home/ycyoon/work/RGTNet')  # Add RGTNet path
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.model_base import ModelBase
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import RGTNet modules
try:
    from model import create_model, load_checkpoint
    from config import setup_args
    RGTNET_AVAILABLE = True
    print("✅ RGTNet modules imported successfully")
except ImportError as e:
    print(f"⚠️ RGTNet modules not available: {e}")
    RGTNET_AVAILABLE = False

class RGTNetModel(ModelBase):
    """Wrapper for locally trained RGTNet models"""
    
    def __init__(self, model_path: str, pretrained_model_name: str = None):
        super().__init__()
        self.model_path = model_path
        self.pretrained_model_name = pretrained_model_name or "meta-llama/Llama-3.2-3B-Instruct"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Loading RGTNet model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model configuration
        self._setup_model_config()
        
        # Load the trained model
        self.model = self._load_trained_model()
        self.model.eval()
        
        print(f"✅ RGTNet model loaded successfully on {self.device}")
    
    def _setup_model_config(self):
        """Setup model configuration based on pretrained model"""
        from transformers import AutoConfig
        
        # Get pretrained model config
        pretrained_config = AutoConfig.from_pretrained(self.pretrained_model_name)
        
        # Create args object for model creation
        class ModelArgs:
            def __init__(self):
                self.d_model = pretrained_config.hidden_size
                self.nhead = pretrained_config.num_attention_heads
                self.num_layers = pretrained_config.num_hidden_layers
                self.dropout = 0.1
                self.max_seq_len = getattr(pretrained_config, 'max_position_embeddings', 2048)
                self.bias_delta = 1.0
                self.vocab_size = pretrained_config.vocab_size
                self.pretrained_model_name = self.pretrained_model_name
        
        self.args = ModelArgs()
        print(f"📊 Model config: d_model={self.args.d_model}, layers={self.args.num_layers}, heads={self.args.nhead}")
    
    def _load_trained_model(self):
        """Load the trained RGTNet model"""
        # Create model
        model = create_model(self.args, self.tokenizer.pad_token_id)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # Handle FSDP checkpoint
            model_state = checkpoint['model_state_dict']
            
            # Try to load state dict
            try:
                model.load_state_dict(model_state, strict=False)
                print("✅ Checkpoint loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint completely: {e}")
                print("🔄 Loading with strict=False for partial compatibility")
                # Load what we can
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in model_state.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
        else:
            # Direct model state
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def generate(self, prompts, **kwargs):
        """Generate responses for given prompts"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.args.max_seq_len - 512,  # Leave room for generation
                    padding=True
                ).to(self.device)
                
                # Generate
                try:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=kwargs.get('max_tokens', 512),
                            temperature=kwargs.get('temperature', 0.7),
                            do_sample=kwargs.get('temperature', 0.7) > 0,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                    
                    # Decode response
                    response = self.tokenizer.decode(
                        outputs[0][inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    responses.append(response)
                    
                except Exception as e:
                    print(f"⚠️ Generation failed for prompt: {e}")
                    responses.append("Error: Generation failed")
        
        return responses if len(responses) > 1 else responses[0]

# First, prepare models and datasets.
TARGET_BASE_URL = "http://localhost:8001/v1"  # Target model (Llama 3.2 3B)
ATTACK_BASE_URL = "http://localhost:8002/v1"  # Attack model (LLama 3.1 70b)
EVAL_BASE_URL = "http://localhost:8003/v1"    # Eval model (HarmBench)
FINAL_EVAL_BASE_URL = "http://localhost:8006/v1"  # Final eval (Meta-Llama-3-70B-Instruct)
REF_EVAL_BASE_URL = "http://localhost:8004/v1"    # Refusal eval (WildGuard)

API_KEY_ATTACK = "EMPTY"  # No API key needed for local vLLM servers
API_KEY_EVAL = "EMPTY"    # No API key needed for local vLLM servers
API_KEY = os.getenv("API_KEY", "EMPTY")

# Foundation models configuration - now for vLLM server usage
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
parser.add_argument('--trained-model-path', type=str, default=None,
                    help='Path to trained RGTNet model checkpoint (e.g., /path/to/model.pth)')
parser.add_argument('--pretrained-model-name', type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                    help='Base pretrained model name for trained model')
parser.add_argument('--use-trained-model', action='store_true',
                    help='Use trained RGTNet model instead of foundation model')
parser.add_argument('--dataset-size', type=int, default=None,
                    help='Number of samples to use from dataset')
args = parser.parse_args()

# Select target model based on argument
model_config = FOUNDATION_MODELS[args.target_model]

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

# Create target model based on arguments
if args.use_trained_model or args.trained_model_path:
    # Use trained RGTNet model
    if not RGTNET_AVAILABLE:
        print("❌ RGTNet modules not available, cannot load trained model")
        sys.exit(1)
    
    if not args.trained_model_path:
        # Default to epoch1 checkpoint if no path specified
        args.trained_model_path = "/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet_epoch1.pth"
    
    if not os.path.exists(args.trained_model_path):
        print(f"❌ Trained model not found at: {args.trained_model_path}")
        print("Available model files:")
        model_dir = os.path.dirname(args.trained_model_path)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(model_dir, f)}")
        sys.exit(1)
    
    print(f"🎯 Using trained RGTNet model: {args.trained_model_path}")
    target_model = RGTNetModel(args.trained_model_path, args.pretrained_model_name)
    target_server_process = None  # No server process for local model
    
else:
    # Use foundation model via vLLM server
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
    
    print(f"✅ Target model {model_config['model_name']} configured for vLLM server!")

# Attack model - still uses vLLM server
attack_model = OpenaiModel(
    model_name="meta-llama/Llama-3.1-70B-Instruct", 
    api_keys=API_KEY_ATTACK, 
    base_url=ATTACK_BASE_URL,
    generation_config={
        "temperature": 1.0,  # 더 높은 temperature로 다양한 응답 유도
        "max_tokens": 2048,  # 줄여서 안정성 확보
        "top_p": 0.95,
        # stop 파라미터 제거 - 너무 일찍 중단될 수 있음
    }
)

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

# Set save path based on model type
if args.use_trained_model or args.trained_model_path:
    model_name_for_save = f"trained-{os.path.basename(args.trained_model_path).replace('.pth', '')}"
else:
    model_name_for_save = args.target_model

SAVE_PATH = f"logs/PAIR-{model_name_for_save}-results"

# Ensure to create the save path
os.makedirs(SAVE_PATH, exist_ok=True)

# PAIR 인스턴스 재생성 (Llama-3.1-70B-Instruct 사용)
attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                final_eval_model=final_eval_model,
                refusal_eval_model=refusal_eval_model,
                eval_model=eval_model,
                n_streams=5,  # Llama-3.1이 더 안정적이므로 원래 값 사용
                n_iterations=5,  # 원래 값으로 복원
                jailbreak_datasets=dataset,
                save_path=SAVE_PATH,
                attack_max_n_tokens=1500,  # 더 많은 토큰 허용
                target_max_n_tokens=1024)

# 공격 실행 (Llama-3.1-70B-Instruct 사용)
try:
    if args.use_trained_model or args.trained_model_path:
        print(f"🎯 Starting PAIR attack on trained RGTNet model: {args.trained_model_path}")
    else:
        print(f"🎯 Starting PAIR attack on {args.target_model} with Llama-3.1-70B-Instruct...")
    
    attacker.attack(save_path=f'{model_name_for_save}_llama31_70b_result.jsonl')
    
    if args.use_trained_model or args.trained_model_path:
        print(f"✅ PAIR attack on trained RGTNet model completed successfully!")
    else:
        print(f"✅ PAIR attack on {args.target_model} completed successfully!")
        
except Exception as e:
    print(f"❌ PAIR attack failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up target server and GPU memory
    if target_server_process:
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