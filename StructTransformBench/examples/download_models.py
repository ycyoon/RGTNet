#!/usr/bin/env python
"""
모델 사전 다운로드 스크립트
HuggingFace 모델을 미리 다운로드하여 캐싱
"""
import os
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# 오프라인 모드 비활성화
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

# 토큰 설정
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("⚠️ HF_TOKEN 환경 변수가 설정되어 있지 않습니다.")
    print("💡 HF_TOKEN을 설정해주세요: export HF_TOKEN=your_token_here")
    sys.exit(1)

def download_model(model_id):
    """지정된 모델 다운로드"""
    print(f"🔄 {model_id} 모델 다운로드 중...")
    try:
        # 스냅샷 다운로드
        snapshot_download(
            repo_id=model_id,
            token=hf_token,
            local_dir=f"./models/{model_id.split('/')[-1]}",
            local_dir_use_symlinks=False
        )
        
        # 토크나이저 다운로드
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        # 모델 가져오기 (파라미터 로드 없이)
        model_info = AutoModelForCausalLM.from_pretrained(
            model_id, 
            token=hf_token,
            device_map=None,  # 메모리에 로드하지 않음
            torch_dtype="auto"
        )
        
        print(f"✅ {model_id} 모델 다운로드 완료")
        return True
    except Exception as e:
        print(f"❌ {model_id} 모델 다운로드 실패: {str(e)}")
        return False

# 다운로드할 모델 목록
models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3.2-1B-Instruct",
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
]

# 모델 다운로드 실행
success_count = 0
for model in models:
    if download_model(model):
        success_count += 1

print(f"\n📊 결과: {success_count}/{len(models)} 모델 다운로드 성공")
