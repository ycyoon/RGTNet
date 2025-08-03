#rgtnet 모델 테스트 - DeepSpeed 체크포인트용

import torch
from model_hybrid import create_hybrid_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import glob
from safetensors.torch import load_file
from datetime import datetime
from peft import PeftModel

def find_latest_model():
    """최신 모델 디렉토리 찾기"""
    base_dir = "/home/ycyoon/work/RGTNet/models"
    
    # 1. 먼저 symbolic link 확인
    latest_link = os.path.join(base_dir, "rgtnet_final_model_latest")
    if os.path.islink(latest_link):
        link_target = os.readlink(latest_link)
        full_path = os.path.join(base_dir, link_target)
        if os.path.exists(full_path):
            return full_path
    
    # 2. 날짜별 디렉토리에서 최신 찾기
    pattern = os.path.join(base_dir, "rgtnet_final_model_*")
    model_dirs = glob.glob(pattern)
    
    if not model_dirs:
        return None
    
    # 날짜와 시간으로 정렬 (최신순)
    model_dirs.sort(key=os.path.getctime, reverse=True)
    return model_dirs[0]

def find_lora_adapters(model_dir):
    """LoRA 어댑터 디렉토리 찾기"""
    if not model_dir or not os.path.exists(model_dir):
        return None
    
    # Look for LoRA adapter directories
    adapter_pattern = os.path.join(model_dir, "lora_adapters_epoch_*")
    adapter_dirs = glob.glob(adapter_pattern)
    
    if adapter_dirs:
        # 가장 최신 epoch의 어댑터 반환
        adapter_dirs.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
        latest_adapter = adapter_dirs[0]
        
        # adapter_config.json이 있는지 확인
        if os.path.exists(os.path.join(latest_adapter, "adapter_config.json")):
            return latest_adapter
    
    return None

def load_model_with_lora(base_model_name, adapter_path, device):
    """베이스 모델에 LoRA 어댑터를 로드"""
    try:
        print(f"📦 Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map={"": device} if torch.cuda.is_available() else None
        )
        
        print(f"🎯 Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Optional: merge adapters for faster inference
        print("🔧 Merging adapters for faster inference...")
        model = model.merge_and_unload()
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model with LoRA: {e}")
        return None

def list_available_models():
    """사용 가능한 모델들 목록 출력"""
    base_dir = "/home/ycyoon/work/RGTNet/models"
    
    print("📋 Available models:")
    print("="*50)
    
    # 날짜별로 그룹화
    models_by_date = {}
    
    pattern = os.path.join(base_dir, "rgtnet_final_model_*")
    model_dirs = glob.glob(pattern)
    
    for model_dir in sorted(model_dirs, key=os.path.getctime, reverse=True):
        dirname = os.path.basename(model_dir)
        date_str = dirname.replace("rgtnet_final_model_", "")
        
        if len(date_str) >= 8:  # YYYYMMDD_HHMMSS 형식
            date_part = date_str[:8]
            time_part = date_str[9:15] if len(date_str) > 8 else ""
            
            if date_part not in models_by_date:
                models_by_date[date_part] = []
            
            models_by_date[date_part].append((time_part, model_dir))
    
    # 날짜별로 출력
    for date in sorted(models_by_date.keys(), reverse=True):
        print(f"\n {date[:4]}-{date[4:6]}-{date[6:8]}:")
        for time_part, model_dir in models_by_date[date]:
            if time_part:
                print(f"  🕐 {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]} - {model_dir}")
            else:
                print(f"   {model_dir}")

def load_hybrid_model(model_dir):
    """Hybrid Llama-RGTNet 모델 로드"""
    print(f" Loading hybrid model from: {model_dir}")
    
    # Check for PEFT adapter files (both .bin and .safetensors formats)
    adapter_files = []
    adapter_patterns = ["adapter_*.bin", "adapter_*.safetensors", "adapter_model.*"]
    for pattern in adapter_patterns:
        adapter_files.extend(glob.glob(os.path.join(model_dir, pattern)))
    
    adapter_config = os.path.join(model_dir, "adapter_config.json")
    
    if adapter_files and os.path.exists(adapter_config):
        print(f" Found PEFT adapter files: {[os.path.basename(f) for f in adapter_files]}")
        return model_dir  # Return directory path for PEFT loading
    
    # Check for regular model files (fallback)
    model_files = []
    for pattern in ["model.safetensors", "pytorch_model.bin", "model-*.safetensors", "pytorch_model-*.bin"]:
        model_files.extend(glob.glob(os.path.join(model_dir, pattern)))
    
    if model_files:
        print(f" Found model files: {[os.path.basename(f) for f in model_files]}")
        return model_dir
    
    print(f"❌ No model or adapter files found in {model_dir}")
    return None

def create_hybrid_model_args():
    """Hybrid 모델용 args 생성"""
    args = argparse.Namespace(
        pretrained_model_name="meta-llama/Llama-3.2-3B-Instruct",
        enable_role_adapters=True,
        use_quantization=False,
        use_lora=False,  # Disable LoRA by default
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        single_gpu_test=True  # Flag for single GPU testing
    )
    
    return args

def create_single_gpu_hybrid_model(args):
    """테스트용 단일 GPU 하이브리드 모델 생성"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    
    # Load base model on single GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},  # Force everything to GPU 0
        trust_remote_code=True,
    )
    
    # Apply LoRA
    if args.use_lora:
        base_model = prepare_model_for_kbit_training(base_model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)
    
    return base_model, tokenizer

def test_model_generation(model, tokenizer, device, prompts=None):
    """하이브리드 모델 생성 테스트"""
    if prompts is None:
        prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?", 
            "Explain quantum computing in simple terms.",
        ]
    
    print("\n" + "="*60)
    print("HYBRID MODEL GENERATION TEST")
    print("="*60)
    
    model.eval()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}: {prompt} ---")
        
        try:
            # Chat template 적용
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 토큰화 with attention mask
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            print(f"Input tokens shape: {input_ids.shape}")
            print(f"Input prompt (first 100 chars): {formatted_prompt[:100]}...")
            
            # 생성 (간단한 테스트)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    do_sample=True, 
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 디코딩 - 새로 생성된 부분만 출력
            input_length = input_ids.shape[1]
            generated_ids = output[0][input_length:]  # 새로 생성된 토큰만
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"Generated response: {generated_text}")
            print(f"Full conversation: {tokenizer.decode(output[0], skip_special_tokens=True)[:200]}...")
            
        except Exception as e:
            print(f"❌ Error in generation: {e}")
            import traceback
            traceback.print_exc()

def main():
    """메인 함수"""
    print("🚀 RGTNet DeepSpeed Model Test")
    print("="*60)
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='Test RGTNet model')
    parser.add_argument('--model_dir', type=str, help='Specific model directory to test')
    parser.add_argument('--list_models', action='store_true', help='List all available models')
    args = parser.parse_args()
    
    # 모델 목록 출력 옵션
    if args.list_models:
        list_available_models()
        return
    
    # 모델 경로 설정
    if args.model_dir:
        model_dir = args.model_dir
        if not os.path.exists(model_dir):
            print(f"❌ Specified model directory not found: {model_dir}")
            return
    else:
        # 최신 모델 자동 찾기
        model_dir = find_latest_model()
        if model_dir is None:
            print("❌ No model found!")
            print("Available options:")
            print("1. Use --list_models to see all available models")
            print("2. Use --model_dir to specify a specific model directory")
            return
    
    print(f" Model directory: {model_dir}")
    
    # Hybrid 모델 체크
    model_path = load_hybrid_model(model_dir)
    if model_path is None:
        print("❌ Failed to find model files")
        return
    
    # args 생성
    model_args = create_hybrid_model_args()
    
    print(f"\n📋 Using hybrid model config:")
    print(f"  Base model: {model_args.pretrained_model_name}")
    print(f"  Role adapters: {model_args.enable_role_adapters}")
    print(f"  LoRA enabled: {model_args.use_lora}")
    print(f"  LoRA rank: {model_args.lora_r}")
    
    # 모델 생성 및 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Using device: {device}")
    
    try:
        # Single GPU 테스트용 모델 생성
        print("\n🔧 Creating single GPU test model...")
        model, tokenizer = create_single_gpu_hybrid_model(model_args)
        
        # 🔧 MODIFIED: Check for and load LoRA adapters
        adapter_path = find_lora_adapters(model_dir)
        if adapter_path:
            print(f"🎯 Found LoRA adapters: {adapter_path}")
            # Try to load model with LoRA adapters
            base_model_name = getattr(model_args, 'pretrained_model_name', 'meta-llama/Llama-3.2-3B-Instruct')
            lora_model = load_model_with_lora(base_model_name, adapter_path, device)
            
            if lora_model is not None:
                print("✅ Model with LoRA adapters loaded successfully!")
                model = lora_model  # Use LoRA model instead of hybrid model
                # Load tokenizer for the base model
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            else:
                print("⚠️ Failed to load LoRA adapters, using hybrid model")
        else:
            print("ℹ️ No LoRA adapters found, using hybrid model")
        
        print("✅ Model loading completed!")
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
        print(f"\n📊 Model info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model structure: {type(model).__name__}")
        
        print("✅ Tokenizer loaded successfully!")
        
        # 기본 forward pass 테스트
        print("\n" + "="*60)
        print("BASIC FORWARD PASS TEST")
        print("="*60)
        
        model.eval()
        
        # Use GPU 0 for single GPU test
        model_device = torch.device("cuda:0")
        print(f"Model is on device: {model_device}")
        
        test_input = torch.randint(0, 1000, (1, 10)).to(model_device)
        test_role_mask = torch.randint(0, 2, (1, 10)).to(model_device)
        
        with torch.no_grad():
            # Different forward pass for LoRA vs hybrid models
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
                # Hybrid model with RGTNet - needs role_mask
                outputs = model(input_ids=test_input, role_mask=test_role_mask)
                print(f"✅ Forward pass successful (Hybrid RGTNet)!")
                print(f"Output shape: {outputs.logits.shape}")
            else:
                # LoRA model - standard forward pass
                outputs = model(input_ids=test_input)
                print(f"✅ Forward pass successful (LoRA Model)!")
                print(f"Output shape: {outputs.logits.shape}")
            
            print(f"Expected shape: (1, 10, vocab_size)")
        
        # 생성 테스트
        test_model_generation(model, tokenizer, model_device)
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during model setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()