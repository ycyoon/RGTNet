# Using Trained RGTNet Models in Benchmarks

이 가이드는 직접 학습한 RGTNet 모델을 jailbreak 공격 벤치마크에서 target model로 사용하는 방법을 설명합니다.

## 지원되는 벤치마크

1. **PAIR Attack** (`run_PAIR.py`)
2. **Multi-Model Benchmark** (`multi_model_benchmark.py`)

## 사용법

### 1. PAIR Attack with Trained Model

#### 기본 사용법
```bash
# 기본 epoch1 모델 사용
python run_PAIR.py --use-trained-model --dataset-size 10

# 특정 모델 경로 지정
python run_PAIR.py --use-trained-model \
    --trained-model-path /home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet_epoch1.pth \
    --dataset-size 10

# 다른 베이스 모델 사용
python run_PAIR.py --use-trained-model \
    --trained-model-path /path/to/your/model.pth \
    --pretrained-model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset-size 10
```

#### 예시 명령어
```bash
# Epoch 1 모델로 PAIR 공격 테스트
python run_PAIR.py --use-trained-model --dataset-size 5

# 파운데이션 모델과 학습된 모델 비교
python run_PAIR.py --target-model llama-3.2-3b --dataset-size 5
python run_PAIR.py --use-trained-model --dataset-size 5
```

### 2. Multi-Model Benchmark with Trained Models

#### 기본 사용법
```bash
# 학습된 모델 추가
python multi_model_benchmark.py \
    --trained-model rgtnet-epoch1 /home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet_epoch1.pth \
    --use-local

# 여러 학습된 모델 추가
python multi_model_benchmark.py \
    --trained-model rgtnet-epoch1 /home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet_epoch1.pth \
    --trained-model rgtnet-final /home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet.pth \
    --use-local
```

### 3. 자동화된 벤치마크 실행

#### run_hf_benchmark.sh 사용
```bash
# 스크립트가 자동으로 학습된 모델을 찾아서 벤치마크 실행
bash run_hf_benchmark.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
- Foundation model 벤치마크
- 학습된 RGTNet 모델 벤치마크 (모델이 존재하는 경우)
- PAIR 공격 (Foundation model + 학습된 모델)

## 모델 경로

기본적으로 다음 경로에서 학습된 모델을 찾습니다:

- `/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet_epoch1.pth`
- `/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet.pth`

## 결과 파일

### PAIR Attack 결과
- Foundation model: `{model_name}_llama31_70b_result.jsonl`
- Trained model: `trained-{model_name}_llama31_70b_result.jsonl`

### Multi-Model Benchmark 결과
- 결과는 `logs/multi_model_benchmark_{timestamp}/` 디렉토리에 저장됩니다.
- 학습된 모델은 `RGTNet-{model_name}` 형태로 표시됩니다.

## 요구사항

1. **RGTNet 모듈**: `/home/ycyoon/work/RGTNet`이 Python path에 있어야 합니다.
2. **모델 파일**: 학습된 `.pth` 파일이 존재해야 합니다.
3. **의존성**: easyjailbreak, transformers, torch 등이 설치되어 있어야 합니다.

## 문제해결

### 모델 로딩 실패
```
❌ Failed to load checkpoint completely: size mismatch...
```
- 체크포인트와 현재 모델 구조가 맞지 않는 경우
- `strict=False`로 부분적 로딩을 시도합니다.

### RGTNet 모듈 없음
```
⚠️ RGTNet modules not available
```
- `/home/ycyoon/work/RGTNet`이 Python path에 있는지 확인
- `model.py`, `config.py` 등 필요한 파일들이 있는지 확인

### GPU 메모리 부족
- 작은 배치 크기 사용
- `--dataset-size`를 줄여서 테스트

## 성능 비교

학습된 모델과 파운데이션 모델의 jailbreak 저항성을 비교할 수 있습니다:

1. **ASR (Attack Success Rate)**: 공격 성공률
2. **Response Quality**: 응답 품질
3. **Refusal Rate**: 거부 응답 비율

결과를 통해 RGTNet 학습이 모델의 안전성에 미치는 영향을 분석할 수 있습니다.
