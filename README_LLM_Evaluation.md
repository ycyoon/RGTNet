# LLM Performance Evaluation Tools

이 도구들은 학습된 LLM 모델의 성능을 평가하기 위한 포괄적인 평가 시스템입니다. BLEU, ROUGE, METEOR, BERTScore 등의 메트릭을 사용하여 골든스탠다드 데이터셋에서 모델 성능을 측정합니다.

## 주요 기능

- **다양한 모델 지원**: RGTNet 모델과 기존 파운데이션 모델 모두 평가 가능
- **포괄적인 메트릭**: BLEU, ROUGE, METEOR, BERTScore 등
- **추가 분석**: 텍스트 길이, 다양성, 유창성 메트릭
- **시각화**: 성능 비교 차트 및 분포 분석
- **다중 모델 비교**: 여러 모델을 한 번에 평가하고 비교

## 설치된 도구들

### 1. 기본 평가 도구
- `evaluate_llm_performance.py`: 기본 LLM 성능 평가
- `evaluate_llm.sh`: 기본 평가 실행 스크립트

### 2. 고급 평가 도구
- `evaluate_llm_advanced.py`: 추가 메트릭과 시각화 포함
- `evaluate_llm_advanced.sh`: 고급 평가 실행 스크립트

### 3. 다중 모델 평가 도구
- `evaluate_multiple_models.py`: 여러 모델 동시 평가 및 비교
- `evaluate_multiple_models.sh`: 다중 모델 평가 실행 스크립트

## 사용법

### 1. 단일 모델 평가 (기본)

```bash
# 기본 평가 실행
./evaluate_llm.sh

# 또는 직접 실행
python evaluate_llm_performance.py \
    --model_path "models" \
    --data_file "data/val_instruction.jsonl" \
    --max_samples 100
```

### 2. 단일 모델 평가 (고급)

```bash
# 고급 평가 실행 (시각화 포함)
./evaluate_llm_advanced.sh

# 또는 직접 실행
python evaluate_llm_advanced.py \
    --model_path "models" \
    --data_file "data/val_instruction.jsonl" \
    --max_samples 200
```

### 3. 다중 모델 평가

```bash
# 설정 파일 생성
python evaluate_multiple_models.py --create_sample_config

# 설정 파일 편집 후 다중 모델 평가 실행
./evaluate_multiple_models.sh
```

## 설정 파일 (model_config.json)

다중 모델 평가를 위한 설정 파일 예시:

```json
{
  "models": [
    {
      "name": "RGTNet_Trained",
      "path": "models",
      "description": "학습된 RGTNet 모델"
    },
    {
      "name": "Llama_3.2_1B_Base",
      "path": "meta-llama/Llama-3.2-1B-Instruct",
      "description": "기본 Llama 3.2 1B Instruct 모델"
    },
    {
      "name": "Llama_3.2_3B_Base",
      "path": "meta-llama/Llama-3.2-3B-Instruct",
      "description": "기본 Llama 3.2 3B Instruct 모델"
    }
  ]
}
```

## 평가 메트릭

### 표준 메트릭
- **BLEU**: Bilingual Evaluation Understudy (문장 및 코퍼스 레벨)
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **BERTScore**: BERT-based evaluation metric (Precision, Recall, F1)

### 추가 메트릭
- **길이 메트릭**: 평균 길이, 표준편차, 최소/최대 길이
- **다양성 메트릭**: Type-Token Ratio, Simpson's Diversity Index
- **유창성 메트릭**: 평균 문장 수, 문장당 평균 단어 수

## 출력 파일

### 기본 평가
- `evaluation_results/llm_evaluation_YYYYMMDD_HHMMSS.json`: 상세 결과
- `evaluation_results/llm_evaluation_YYYYMMDD_HHMMSS_summary.csv`: 요약 결과

### 고급 평가
- `evaluation_results/advanced_llm_evaluation_YYYYMMDD_HHMMSS.json`: 상세 결과
- `evaluation_results/advanced_llm_evaluation_YYYYMMDD_HHMMSS_summary.csv`: 요약 결과
- `evaluation_results/advanced_llm_evaluation_YYYYMMDD_HHMMSS_visualizations/`: 시각화 파일들

### 다중 모델 평가
- `evaluation_results/model_comparison_YYYYMMDD_HHMMSS.csv`: 모델 비교 결과
- `evaluation_results/advanced_model_comparison_YYYYMMDD_HHMMSS.csv`: 고급 비교 결과

## 모델 지원

### RGTNet 모델
- 학습된 RGTNet 모델을 자동으로 감지하고 로드
- 토크나이저는 모델 경로에서 자동 로드

### 파운데이션 모델
- Hugging Face의 모든 CausalLM 모델 지원
- 예: Llama, Mistral, GPT-2, GPT-Neo 등
- 토크나이저는 모델 경로에서 자동 로드

## 주의사항

1. **토크나이저**: 모든 모델에서 토크나이저는 모델 경로와 동일한 경로에서 로드됩니다.
2. **메모리**: 대용량 모델 평가 시 충분한 GPU 메모리가 필요합니다.
3. **시간**: BERTScore 계산은 시간이 오래 걸릴 수 있습니다.
4. **데이터셋**: JSONL 형식의 instruction-response 쌍이 필요합니다.

## 예시 실행

```bash
# 1. 기본 평가
./evaluate_llm.sh

# 2. 고급 평가 (시각화 포함)
./evaluate_llm_advanced.sh

# 3. 다중 모델 비교
python evaluate_multiple_models.py --create_sample_config
# model_config.json 편집 후
./evaluate_multiple_models.sh
```

## 결과 해석

- **BLEU**: 0-1 범위, 높을수록 좋음
- **ROUGE**: 0-1 범위, 높을수록 좋음
- **METEOR**: 0-1 범위, 높을수록 좋음
- **BERTScore**: 0-1 범위, 높을수록 좋음
- **길이 차이**: 참조와 예측 간의 길이 차이, 작을수록 좋음
- **다양성**: Type-Token Ratio와 Simpson Diversity, 참조와 유사할수록 좋음
