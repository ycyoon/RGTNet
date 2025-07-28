#!/usr/bin/env python3
"""
모델 로더 패치 스크립트
======================

기존 스크립트들을 수정하지 않고도 오프라인 모드에서 안전하게 모델을 로드할 수 있도록
transformers 라이브러리의 모델 로딩 함수를 패치합니다.

사용법:
    import patch_model_loader  # 스크립트 최상단에 이 한 줄만 추가
    
    # 이제 기존 코드가 모두 오프라인에서 작동함
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
"""
import os
import warnings
from functools import wraps

# 오프라인 모드 강제 설정
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 캐시 디렉토리 설정
CACHE_DIR = "/ceph_data/ycyoon/.cache/huggingface"
if os.path.exists(CACHE_DIR):
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/transformers"

def make_offline_safe(original_func):
    """모델 로딩 함수를 오프라인 안전 버전으로 만드는 데코레이터"""
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # local_files_only가 명시적으로 False로 설정되지 않은 경우 True로 설정
        if 'local_files_only' not in kwargs:
            kwargs['local_files_only'] = True
        
        # trust_remote_code 안전성 설정
        if 'trust_remote_code' not in kwargs:
            kwargs['trust_remote_code'] = False
            
        try:
            return original_func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in ['offline mode', 'cannot reach', 'connection', 'network']):
                print(f"⚠️ 온라인 접근 시도 감지: {e}")
                print("🔒 오프라인 모드로 재시도 중...")
                kwargs['local_files_only'] = True
                return original_func(*args, **kwargs)
            else:
                raise
    return wrapper

# transformers 라이브러리 패치
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
    
    # 원본 함수들 백업
    _original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
    _original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained
    _original_config_from_pretrained = AutoConfig.from_pretrained
    _original_auto_model_from_pretrained = AutoModel.from_pretrained
    
    # 패치된 함수들로 교체
    AutoModelForCausalLM.from_pretrained = staticmethod(make_offline_safe(_original_model_from_pretrained))
    AutoTokenizer.from_pretrained = staticmethod(make_offline_safe(_original_tokenizer_from_pretrained))
    AutoConfig.from_pretrained = staticmethod(make_offline_safe(_original_config_from_pretrained))
    AutoModel.from_pretrained = staticmethod(make_offline_safe(_original_auto_model_from_pretrained))
    
    print("🔒 모델 로더 오프라인 패치 적용 완료")
    print("📁 캐시 디렉토리:", os.environ.get("HF_HOME", "default"))
    
except ImportError as e:
    warnings.warn(f"⚠️ transformers 라이브러리 패치 실패: {e}")

# 추가 안전장치: huggingface_hub 패치
try:
    from huggingface_hub import snapshot_download, hf_hub_download
    
    _original_snapshot_download = snapshot_download
    _original_hf_hub_download = hf_hub_download
    
    def safe_snapshot_download(*args, **kwargs):
        if 'local_files_only' not in kwargs:
            kwargs['local_files_only'] = True
        return _original_snapshot_download(*args, **kwargs)
    
    def safe_hf_hub_download(*args, **kwargs):
        if 'local_files_only' not in kwargs:
            kwargs['local_files_only'] = True
        return _original_hf_hub_download(*args, **kwargs)
    
    # 함수 교체
    import huggingface_hub
    huggingface_hub.snapshot_download = safe_snapshot_download
    huggingface_hub.hf_hub_download = safe_hf_hub_download
    
    print("🔒 huggingface_hub 오프라인 패치 적용 완료")
    
except ImportError:
    pass  # huggingface_hub가 없으면 무시

print("✅ 모든 모델 로딩이 오프라인 모드로 설정되었습니다!")
