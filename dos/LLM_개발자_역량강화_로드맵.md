# LLM 개발자 역량 강화 로드맵

> RAG 엔지니어에서 LLM 연구/개발자로의 전환 가이드

---

## Executive Summary

현재 RAG 파이프라인 구축, 프롬프트 엔지니어링, API 활용 수준에서 **파인튜닝, 모델 내부 이해, 연구/개발 역량**으로 레벨업하기 위한 체계적인 로드맵입니다.

### 현재 상황 vs 목표 역량

| 현재 역량 | 목표 역량 |
|-----------|-----------|
| GPT API를 활용한 RAG 시스템 구축 | 오픈소스 LLM 파인튜닝 및 커스터마이징 |
| 문서 청킹, 임베딩, 벡터 DB 활용 | LoRA/QLoRA를 활용한 효율적 학습 |
| 프롬프트 엔지니어링 | DPO/RLHF를 통한 모델 정렬(Alignment) |
| Cursor/Claude Code로 개발 | Transformer 아키텍처 깊은 이해 |
| DB 데이터 분석 및 서머리 | LLM 평가 및 벤치마킹 역량 |
| - | 모델 추론 최적화 및 프로덕션 배포 |

---

## Phase 1: 기초 다지기 (4-6주)

이 단계에서는 파인튜닝에 필요한 이론적 기반을 확립합니다.

### 1.1 Transformer 아키텍처 심층 학습

#### 핵심 학습 자료

- **Andrej Karpathy - Neural Networks: Zero to Hero**
  - 🔗 https://karpathy.ai/zero-to-hero.html
  - 특히 'Let's Build GPT from Scratch' 영상은 필수. 2시간 안에 전체 GPT 아키텍처를 처음부터 구현하며 Self-Attention, Multi-Head Attention을 완벽히 이해할 수 있음

- **Jay Alammar - The Illustrated Transformer**
  - 🔗 https://jalammar.github.io/illustrated-transformer/

#### 실습 프로젝트: nanoGPT 구현

Karpathy의 nanoGPT를 직접 따라 구현:
- 🔗 https://github.com/karpathy/nanoGPT

```
목표: Shakespeare 텍스트로 소규모 GPT 훈련
학습 포인트: Tokenization, Attention 메커니즘, Position Encoding
```

### 1.2 개발 환경 구축

#### 필수 도구

| 도구 | 용도 | 설치 |
|------|------|------|
| Hugging Face | 모델/데이터셋 허브, Transformers 라이브러리 | `pip install transformers datasets` |
| Unsloth | 2x 빠른 파인튜닝, 70% 메모리 절약 | `pip install unsloth` |
| TRL | SFT, DPO, RLHF 트레이너 | `pip install trl` |
| PEFT | LoRA, QLoRA 등 효율적 파인튜닝 | `pip install peft` |
| bitsandbytes | 4-bit/8-bit 양자화 | `pip install bitsandbytes` |
| Weights & Biases | 실험 추적 및 시각화 | `pip install wandb` |

#### GPU 환경

- **최소 요구사항:** RTX 3090/4090 (24GB VRAM) 또는 A10G
- **무료 옵션:** Google Colab (T4 GPU), Kaggle (P100/T4)
- **QLoRA 사용 시** 7B 모델을 6.5GB VRAM에서 파인튜닝 가능

### ✅ Phase 1 체크포인트

스스로 점검해보세요:

- [ ] Self-Attention의 Q, K, V가 무엇이고 왜 필요한지 설명할 수 있는가?
- [ ] Multi-Head Attention이 Single-Head보다 좋은 이유를 설명할 수 있는가?
- [ ] Position Encoding이 왜 필요한지, 어떻게 동작하는지 아는가?
- [ ] nanoGPT 코드에서 Attention 계산 부분을 찾아 설명할 수 있는가?
- [ ] Transformer의 Encoder와 Decoder 차이를 설명할 수 있는가?

---

## Phase 2: Hugging Face 생태계 입문 (2-3주)

> ⚠️ **중요:** nanoGPT에서 바로 파인튜닝으로 넘어가면 간극이 큽니다. 이 단계에서 실무 도구에 익숙해지세요.

### 2.1 Transformers 라이브러리 기초

#### 핵심 학습 내용

```python
# 이 코드들이 자연스러워질 때까지 연습
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 토크나이저 이해
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokens = tokenizer("Hello, world!", return_tensors="pt")
# input_ids, attention_mask 이해하기

# 2. 모델 로드 및 추론
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. 텍스트 생성
outputs = model.generate(**tokens, max_new_tokens=100)
```

#### 필수 실습

| 실습 | 학습 포인트 |
|------|-------------|
| 다양한 모델 로드해보기 | model card 읽기, 라이선스 확인, 요구 VRAM 파악 |
| Tokenizer 실험 | BPE vs SentencePiece, special tokens, chat template |
| 생성 파라미터 실험 | temperature, top_p, top_k, repetition_penalty |
| Pipeline API 사용 | 간단한 텍스트 분류, 요약, QA 태스크 |

#### 학습 자료

- 🔗 [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) - 무료, 체계적
- 🔗 [Transformers 공식 문서](https://huggingface.co/docs/transformers)

### 2.2 Datasets 라이브러리

```python
from datasets import load_dataset, Dataset

# 공개 데이터셋 로드
dataset = load_dataset("tatsu-lab/alpaca")

# 커스텀 데이터셋 생성
my_data = Dataset.from_dict({
    "instruction": [...],
    "input": [...],
    "output": [...]
})

# 데이터 전처리
def formatting_func(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": formatting_func(x)})
```

### ✅ Phase 2 체크포인트

- [ ] 원하는 모델을 HuggingFace에서 찾아 로드할 수 있는가?
- [ ] Tokenizer의 special tokens (BOS, EOS, PAD)를 설명할 수 있는가?
- [ ] Chat template이 무엇이고 왜 중요한지 아는가?
- [ ] 데이터셋을 로드하고 전처리할 수 있는가?
- [ ] `device_map="auto"`와 `torch_dtype`의 역할을 아는가?

---

## Phase 3: 파인튜닝 데이터 준비 (2-3주)

> ⚠️ **핵심 원칙:** 파인튜닝 성공의 80%는 데이터 품질입니다.

### 3.1 Instruction 데이터 형식 이해

#### 주요 데이터 형식

| 형식 | 구조 | 용도 |
|------|------|------|
| **Alpaca** | instruction, input, output | 단일 턴 instruction following |
| **ShareGPT** | conversations: [{from, value}, ...] | 멀티턴 대화 |
| **ChatML** | `<\|im_start\|>role\n content<\|im_end\|>` | OpenAI 스타일 대화 |
| **Llama 3 Format** | `<\|begin_of_text\|><\|start_header_id\|>...` | Llama 3 전용 |

#### 형식 선택 가이드

```
단일 턴 QA/작업 → Alpaca 형식
멀티턴 대화 → ShareGPT 형식
특정 모델 파인튜닝 → 해당 모델의 chat template 사용
```

### 3.2 데이터 품질 관리

#### 품질 체크리스트

```
□ 중복 데이터 제거 (exact match + near-duplicate)
□ 너무 짧거나 긴 샘플 필터링
□ 언어 일관성 확인 (한국어 데이터에 영어 섞임 등)
□ 형식 일관성 검증 (JSON 파싱 에러 등)
□ 유해/부적절 콘텐츠 필터링
□ 라벨 품질 검증 (랜덤 샘플링 후 수동 검토)
```

#### 데이터 정제 코드 예시

```python
def clean_dataset(dataset):
    # 1. 길이 필터링
    dataset = dataset.filter(lambda x: 50 < len(x['output']) < 2000)

    # 2. 중복 제거
    seen = set()
    def remove_duplicates(example):
        key = example['instruction'][:100]
        if key in seen:
            return False
        seen.add(key)
        return True
    dataset = dataset.filter(remove_duplicates)

    # 3. 품질 점수 기반 필터링 (선택적)
    # GPT-4로 품질 점수 매기고 threshold 이상만 사용

    return dataset
```

### 3.3 합성 데이터 생성

실제 데이터가 부족할 때 LLM을 활용한 데이터 생성:

#### Self-Instruct 방식

```python
# GPT-4/Claude를 활용한 instruction 생성
prompt = """
다음은 SQL 쿼리 작성 태스크의 예시입니다:

예시 1:
- 질문: 모든 직원의 이름과 급여를 조회하세요
- SQL: SELECT name, salary FROM employees

위 예시를 참고하여 새로운 SQL 관련 질문-답변 쌍을 5개 생성하세요.
다양한 난이도와 SQL 기능(JOIN, GROUP BY, 서브쿼리 등)을 포함하세요.
"""
```

#### Evol-Instruct 방식 (WizardLM)

```
기존 instruction을 점진적으로 복잡하게 변형:
1. 제약 조건 추가
2. 추론 단계 심화
3. 구체적인 입력 추가
4. 복합 작업으로 확장
```

### 3.4 데이터 오염(Contamination) 방지

벤치마크 테스트셋이 학습 데이터에 포함되면 평가가 무의미해집니다.

```
주의사항:
□ MMLU, HumanEval 등 벤치마크 데이터 제외
□ 공개 테스트셋과의 n-gram 오버랩 체크
□ 웹 크롤링 데이터 사용 시 특히 주의
□ 합성 데이터 생성 시 벤치마크 문제 참조 금지
```

### ✅ Phase 3 체크포인트

- [ ] Alpaca, ShareGPT, ChatML 형식의 차이를 설명할 수 있는가?
- [ ] 사용할 모델의 chat template을 적용할 수 있는가?
- [ ] 데이터 품질 이슈를 식별하고 정제할 수 있는가?
- [ ] Self-Instruct로 합성 데이터를 생성할 수 있는가?
- [ ] 데이터 오염이 왜 문제인지 설명할 수 있는가?

---

## Phase 4: LoRA/QLoRA 파인튜닝 실습 (4-6주)

실제 데이터셋으로 오픈소스 LLM을 파인튜닝하는 핵심 역량을 기릅니다.

### 4.1 LoRA/QLoRA 이론

#### 핵심 개념

**LoRA (Low-Rank Adaptation)** 는 사전학습된 모델의 가중치를 직접 수정하는 대신, 작은 저차원(low-rank) 행렬 A와 B를 추가하여 훈련합니다. 전체 파라미터의 약 **1%만 학습**하면서도 전체 파인튜닝에 근접한 성능을 달성합니다.

**QLoRA**는 LoRA에 4-bit 양자화를 결합하여 메모리 사용량을 추가로 **75% 절감**합니다. 65B 파라미터 모델도 단일 48GB GPU에서 파인튜닝이 가능해집니다.

#### 핵심 하이퍼파라미터

| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| r (rank) | 8, 16, 32, 64 | 저차원 행렬의 랭크. 높을수록 표현력↑, 메모리↑ |
| alpha | r의 2배 (예: r=16 → alpha=32) | 스케일링 팩터. alpha/r 비율이 학습률에 영향 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | LoRA를 적용할 레이어들 |
| learning_rate | 1e-5 ~ 5e-5 | 파인튜닝 시 낮은 학습률 사용 권장 |

### 4.2 실습 프로젝트 #1: Text-to-SQL 파인튜닝

자연어 질문을 SQL 쿼리로 변환하는 모델을 파인튜닝합니다.

#### 데이터셋

| 데이터셋 | 크기 | 링크 |
|----------|------|------|
| b-mc2/sql-create-context | 78K 예제 | [Hugging Face](https://huggingface.co/datasets/b-mc2/sql-create-context) |
| Clinton/Text-to-sql-v1 | 262K 예제 | [Hugging Face](https://huggingface.co/datasets/Clinton/Text-to-sql-v1) |
| gretelai/synthetic_text_to_sql | 100K 합성 데이터 | [Hugging Face](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) |

#### 권장 베이스 모델

- **Llama 3.1 8B Instruct** - 범용성과 한국어 지원
- **Mistral 7B Instruct** - 효율적이고 빠른 추론
- **Qwen 2.5 7B** - 코드 및 수학 능력 우수

#### 참고 튜토리얼

🔗 [Phil Schmid: How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl)

### 4.3 실습 프로젝트 #2: 한국어 도메인 특화 모델

#### 한국어 데이터셋

| 데이터셋 | 용도 | 링크 |
|----------|------|------|
| nlpai-lab/kullm-v2 | 한국어 instruction | [Hugging Face](https://huggingface.co/datasets/nlpai-lab/kullm-v2) |
| maywell/ko_wikidata_QA | 한국어 QA | [Hugging Face](https://huggingface.co/datasets/maywell/ko_wikidata_QA) |
| beomi/KoAlpaca-v1.1a | 한국어 Alpaca 스타일 | [Hugging Face](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a) |
| AI Hub 공공데이터 | 다양한 도메인 데이터 | [aihub.or.kr](https://aihub.or.kr) |

#### 추천 도메인 프로젝트

1. **법률 상담 챗봇:** 법률 QA 데이터로 파인튜닝
2. **의료 정보 어시스턴트:** 의학 용어/증상 데이터 활용
3. **금융 분석 도우미:** 재무제표, 투자 용어 특화

### 4.4 Unsloth 활용 빠른 파인튜닝

Unsloth는 HuggingFace Transformers 대비 **2배 빠른 학습 속도**와 **70% 적은 메모리** 사용량을 제공합니다.

#### 핵심 기능

- 9GB VRAM으로 Llama 3.1 8B 파인튜닝 가능
- GGUF, vLLM 형식으로 쉬운 배포
- Llama 4, Qwen3, DeepSeek 등 최신 모델 지원
- GRPO 기반 강화학습(RL) 지원

🔗 [공식 문서](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
🔗 [GitHub](https://github.com/unslothai/unsloth)

### ✅ Phase 4 체크포인트

- [ ] LoRA가 왜 효율적인지 수학적으로 설명할 수 있는가?
- [ ] r, alpha 값을 변경했을 때 어떤 영향이 있는지 아는가?
- [ ] QLoRA의 4-bit NormalFloat가 무엇인지 설명할 수 있는가?
- [ ] 직접 파인튜닝한 모델을 HuggingFace에 업로드했는가?
- [ ] 학습 loss 그래프를 해석할 수 있는가?

---

## Phase 5: 학습 트러블슈팅 가이드

> ⚠️ **현실:** 파인튜닝은 한 번에 성공하지 않습니다. 문제 해결 능력이 핵심입니다.

### 5.1 일반적인 문제와 해결책

#### Loss가 떨어지지 않을 때

| 증상 | 가능한 원인 | 해결책 |
|------|-------------|--------|
| Loss가 전혀 안 떨어짐 | 학습률이 너무 낮음 | lr을 10배 높여보기 |
| Loss가 튀거나 NaN 발생 | 학습률이 너무 높음 | lr을 10배 낮추기, gradient clipping 적용 |
| Loss가 초반만 떨어지고 정체 | 데이터 품질 문제 | 데이터 다시 검토, 더 다양한 데이터 추가 |
| Loss가 다시 올라감 (과적합) | epoch이 너무 많음 | early stopping, 데이터 증강 |

#### 메모리 부족 (OOM) 해결

```python
# 1. Gradient Checkpointing 활성화
model.gradient_checkpointing_enable()

# 2. 배치 사이즈 줄이고 Gradient Accumulation 사용
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 최소값
    gradient_accumulation_steps=16,  # effective batch = 16
)

# 3. 더 공격적인 양자화
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # 이중 양자화
)

# 4. LoRA rank 줄이기
lora_config = LoraConfig(r=8)  # 16 → 8

# 5. Max sequence length 줄이기
max_seq_length = 1024  # 2048 → 1024
```

### 5.2 학습 불안정성 디버깅

#### 체크리스트

```
□ 데이터에 빈 문자열이나 None이 있는지 확인
□ 토큰화 후 시퀀스 길이 분포 확인
□ special token이 제대로 설정되었는지 확인
□ padding side가 모델에 맞는지 확인 (보통 left for generation)
□ BOS/EOS 토큰이 중복 추가되지 않았는지 확인
```

#### Wandb로 디버깅하기

```python
import wandb

# 학습 중 추적할 지표
wandb.log({
    "train/loss": loss,
    "train/learning_rate": lr,
    "train/grad_norm": grad_norm,  # gradient explosion 감지
    "eval/loss": eval_loss,
    "samples/random_output": generated_text,  # 주기적으로 생성 결과 확인
})
```

### 5.3 Catastrophic Forgetting (치명적 망각)

파인튜닝 후 기존 능력이 저하되는 현상:

#### 방지 전략

| 전략 | 설명 |
|------|------|
| **낮은 학습률 사용** | 1e-5 이하로 설정 |
| **적은 epoch** | 1-3 epoch만 학습 |
| **Replay buffer** | 일반 데이터 일부를 학습에 포함 |
| **LoRA rank 제한** | 너무 높은 rank는 과적합 유발 |
| **정기적 평가** | 기존 벤치마크 성능 모니터링 |

### ✅ Phase 5 체크포인트

- [ ] OOM 에러가 발생했을 때 단계별로 대응할 수 있는가?
- [ ] Loss 그래프를 보고 문제를 진단할 수 있는가?
- [ ] Wandb로 실험을 추적하고 비교할 수 있는가?
- [ ] Catastrophic forgetting을 어떻게 방지하는지 설명할 수 있는가?

---

## Phase 6: 모델 정렬(Alignment) - DPO (3-4주)

파인튜닝된 모델을 인간의 선호도에 맞게 정렬하는 고급 기법을 학습합니다.

### 6.1 DPO (Direct Preference Optimization)

#### 왜 DPO인가?

기존 RLHF는 보상 모델 훈련 → PPO 강화학습이라는 복잡한 2단계 과정이 필요했습니다. DPO는 이를 **단일 분류 손실함수**로 단순화하여:

- 별도 보상 모델 훈련 불필요
- PPO 대비 안정적인 훈련
- 구현 및 디버깅 용이

> 💡 **참고:** RLHF(PPO)는 구현 복잡성 대비 DPO와 성능 차이가 크지 않아, 실무에서는 DPO를 주로 사용합니다.

#### Preference 데이터셋

| 데이터셋 | 설명 |
|----------|------|
| argilla/ultrafeedback-binarized-preferences-cleaned | 정제된 UltraFeedback 선호도 데이터 |
| Anthropic/hh-rlhf | Anthropic의 Helpfulness/Harmlessness 데이터 |
| HuggingFaceH4/ultrafeedback_binarized | GPT-4 점수 기반 선호도 쌍 |

#### 실습 프로젝트: DPO로 모델 정렬

🔗 [Phil Schmid - RLHF in 2024 with DPO](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl)

### 6.2 GRPO (Group Relative Policy Optimization)

DeepSeek R1에서 사용된 최신 정렬 기법으로, **추론(reasoning) 능력 강화**에 효과적입니다.

🔗 [Unsloth GRPO 가이드](https://unsloth.ai/blog)

### ✅ Phase 6 체크포인트

- [ ] DPO의 손실 함수를 직관적으로 설명할 수 있는가?
- [ ] Preference 데이터의 구조(chosen, rejected)를 이해하는가?
- [ ] SFT 모델을 DPO로 추가 정렬해본 경험이 있는가?

---

## Phase 7: LLM 평가 및 벤치마킹 (3-4주)

파인튜닝된 모델의 성능을 객관적으로 측정하고 개선하는 역량을 기릅니다.

### 7.1 주요 벤치마크

| 벤치마크 | 측정 항목 | 특징 |
|----------|-----------|------|
| MMLU / MMLU-Pro | 57개 과목 지식 테스트 | 범용 지식 평가의 표준 |
| HumanEval | 164개 코딩 문제 | Pass@k 메트릭으로 코드 정확도 측정 |
| MT-Bench | 멀티턴 대화 품질 | LLM-as-Judge 방식 |
| Chatbot Arena | 인간 선호도 기반 | ELO 레이팅 시스템 |
| GSM8K | 수학 추론 | 초등학교 수준 수학 문제 |

### 7.2 평가 도구

- **lm-evaluation-harness:** Eleuther AI의 표준 평가 프레임워크
  - 🔗 https://github.com/EleutherAI/lm-evaluation-harness

- **DeepEval:** LLM 평가를 위한 Python 프레임워크
  - 🔗 https://deepeval.com

- **Open LLM Leaderboard:** HuggingFace 공식 리더보드
  - 🔗 https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

### 7.3 커스텀 평가셋 구축

공개 벤치마크만으로는 도메인 특화 성능을 측정할 수 없습니다.

#### 평가셋 설계 원칙

```
1. 목적 정의: 무엇을 측정할 것인가?
   - 정확성? 유창성? 안전성? 도메인 지식?

2. 데이터 수집
   - 최소 100-500개 테스트 샘플
   - 다양한 난이도 분포
   - 학습 데이터와 겹치지 않도록 주의

3. 평가 방식 결정
   - 자동 평가: exact match, BLEU, ROUGE, BERTScore
   - LLM-as-Judge: GPT-4로 품질 점수 매기기
   - Human evaluation: 최종 검증용
```

#### LLM-as-Judge 예시

```python
judge_prompt = """
다음 응답의 품질을 1-5점으로 평가하세요.

질문: {question}
응답: {answer}

평가 기준:
- 정확성: 사실에 부합하는가?
- 완전성: 질문에 충분히 답했는가?
- 명확성: 이해하기 쉽게 작성되었는가?

점수와 이유를 JSON 형식으로 출력하세요.
"""
```

### 7.4 A/B 테스트 및 Human Evaluation

#### Human Evaluation 설계

| 평가 유형 | 방식 | 장단점 |
|----------|------|--------|
| **Absolute scoring** | 각 응답을 독립적으로 1-5점 평가 | 간단하지만 평가자 편향 발생 |
| **Pairwise comparison** | 두 응답 중 더 나은 것 선택 | 더 신뢰성 높음, 시간 소요 큼 |
| **Best-of-N** | N개 응답 중 가장 좋은 것 선택 | 여러 모델 동시 비교에 적합 |

#### 평가 가이드라인 예시

```markdown
## 평가자 가이드라인

### 평가 기준
1. **유용성 (40%)**: 질문에 도움이 되는 답변인가?
2. **정확성 (30%)**: 사실적으로 올바른가?
3. **안전성 (20%)**: 유해하거나 부적절한 내용이 없는가?
4. **자연스러움 (10%)**: 읽기 편하고 자연스러운가?

### 주의사항
- 개인적 선호가 아닌 객관적 기준으로 평가
- 확신이 없으면 "판단 불가" 선택 가능
- 각 평가에 20초 이상 소요
```

### ✅ Phase 7 체크포인트

- [ ] lm-evaluation-harness로 모델을 평가해본 경험이 있는가?
- [ ] 도메인 특화 평가셋을 직접 만들 수 있는가?
- [ ] LLM-as-Judge 방식의 장단점을 설명할 수 있는가?
- [ ] Human evaluation을 설계할 수 있는가?
- [ ] 과적합을 평가 결과에서 어떻게 탐지하는지 아는가?

---

## Phase 8: 추론 최적화 및 프로덕션 배포 (4-5주)

모델을 만드는 것에서 끝나지 않습니다. 실제 서비스에 배포하는 역량이 필수입니다.

### 8.1 모델 양자화

#### 양자화 옵션 비교

| 방식 | 정밀도 | 크기 감소 | 품질 손실 | 용도 |
|------|--------|-----------|-----------|------|
| FP16 | 16-bit | 50% | 거의 없음 | 학습/고품질 추론 |
| INT8 | 8-bit | 75% | 매우 적음 | 서버 배포 |
| INT4 (GPTQ/AWQ) | 4-bit | 87.5% | 약간 있음 | 엣지/로컬 배포 |
| GGUF (llama.cpp) | 2-8bit | 다양함 | 설정에 따라 | CPU/로컬 추론 |

#### GGUF 변환 및 Ollama 배포

```bash
# 1. HuggingFace 모델을 GGUF로 변환
python llama.cpp/convert_hf_to_gguf.py \
    ./my-finetuned-model \
    --outfile my-model.gguf \
    --outtype q4_k_m  # 4-bit 양자화

# 2. Ollama에 등록
ollama create my-model -f Modelfile

# 3. 로컬에서 실행
ollama run my-model
```

### 8.2 추론 서버 구축

#### vLLM - 고성능 추론 엔진

```python
from vllm import LLM, SamplingParams

# 모델 로드 (PagedAttention으로 메모리 효율적)
llm = LLM(
    model="my-finetuned-model",
    tensor_parallel_size=1,  # GPU 수
    gpu_memory_utilization=0.9,
)

# 배치 추론
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

#### FastAPI + vLLM 서버

```python
from fastapi import FastAPI
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="my-model")

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 512):
    params = SamplingParams(max_tokens=max_tokens)
    output = llm.generate([prompt], params)[0]
    return {"response": output.outputs[0].text}
```

### 8.3 추론 최적화 기법

| 기법 | 효과 | 구현 복잡도 |
|------|------|-------------|
| **KV-Cache** | 반복 계산 제거, 자동 적용 | 자동 |
| **Continuous Batching** | 처리량 2-10x 향상 | vLLM 사용 |
| **PagedAttention** | 메모리 효율 90%+ | vLLM 사용 |
| **Speculative Decoding** | 지연시간 2-3x 감소 | 중간 |
| **Flash Attention** | 메모리/속도 개선 | 라이브러리 설치 |

### 8.4 배포 옵션

| 옵션 | 장점 | 단점 | 적합한 경우 |
|------|------|------|-------------|
| **Ollama** | 간단, 로컬 테스트 | 확장성 제한 | 개인 사용, 프로토타입 |
| **vLLM + FastAPI** | 고성능, 유연함 | 인프라 관리 필요 | 프로덕션 서비스 |
| **HuggingFace Inference Endpoints** | 관리형, 간편 | 비용 높음 | 빠른 배포 필요 시 |
| **TGI (Text Generation Inference)** | HF 최적화, 안정적 | vLLM보다 느릴 수 있음 | HF 생태계 선호 시 |
| **RunPod / Modal** | 서버리스 GPU | 비용 예측 어려움 | 간헐적 사용 |

### ✅ Phase 8 체크포인트

- [ ] 파인튜닝한 모델을 GGUF로 변환할 수 있는가?
- [ ] Ollama로 로컬에서 모델을 서빙해본 경험이 있는가?
- [ ] vLLM의 장점(PagedAttention 등)을 설명할 수 있는가?
- [ ] FastAPI로 간단한 추론 서버를 구축할 수 있는가?
- [ ] 양자화 방식별 trade-off를 설명할 수 있는가?

---

## Phase 9: 최신 기법 및 심화 학습 (선택, 4-6주)

핵심 역량을 갖춘 후, 최신 연구 동향을 따라가는 단계입니다.

### 9.1 최신 아키텍처

| 기법 | 설명 | 관련 모델 |
|------|------|-----------|
| **MoE (Mixture of Experts)** | 입력에 따라 일부 파라미터만 활성화 | Mixtral, DeepSeek |
| **Mamba / State Space Models** | Transformer 대안, 선형 복잡도 | Mamba, Jamba |
| **Long Context 기법** | RoPE scaling, YaRN, ALiBi | Llama 3 (128K context) |

### 9.2 고급 학습 기법

| 기법 | 설명 | 용도 |
|------|------|------|
| **Continued Pre-training** | 도메인 데이터로 추가 사전학습 | 도메인 적응 (의료, 법률 등) |
| **Model Merging** | 여러 모델의 가중치 병합 | TIES, DARE, SLERP |
| **Distillation** | 큰 모델 → 작은 모델 지식 전이 | 경량화 |
| **Curriculum Learning** | 쉬운 것부터 어려운 순서로 학습 | 복잡한 태스크 학습 |

### 9.3 핵심 논문 읽기

#### 필독 논문

| 논문 | 핵심 내용 |
|------|-----------|
| **Attention Is All You Need** | Transformer 원본, 모든 것의 시작 |
| **LoRA** | 효율적 파인튜닝의 표준 |
| **QLoRA** | 4-bit 양자화 + LoRA |
| **DPO** | 선호도 학습의 단순화 |
| **Scaling Laws** | 모델/데이터 크기와 성능 관계 |
| **Chain-of-Thought Prompting** | 추론 능력 향상 기법 |

### ✅ Phase 9 체크포인트

- [ ] MoE가 왜 효율적인지 설명할 수 있는가?
- [ ] Continued Pre-training과 Fine-tuning의 차이를 아는가?
- [ ] 위 필독 논문 중 3개 이상을 읽고 요약할 수 있는가?

---

## Phase 10: 에이전트 시스템 고도화 (선택 트랙, 4-6주)

> 💡 **참고:** 이 Phase는 핵심 파인튜닝 역량과 별개의 트랙입니다. 관심에 따라 Phase 8 이후 병행하거나 나중에 진행하세요.

### 10.1 파인튜닝 모델 + 에이전트 통합

에이전트의 두뇌로 직접 파인튜닝한 모델을 사용하는 프로젝트:

```python
# 예시: Tool-use 특화 파인튜닝 모델을 에이전트에 적용
from langchain.llms import VLLM
from langchain.agents import create_tool_calling_agent

# 파인튜닝한 모델 로드
llm = VLLM(
    model="my-tool-use-finetuned-model",
    trust_remote_code=True,
)

# 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)
```

### 10.2 LangGraph 마스터하기

LangGraph는 LangChain의 확장으로, **상태 기반 멀티에이전트 워크플로우**를 구축할 수 있습니다.

#### 핵심 개념

- **State Graph:** 상태를 유지하며 실행되는 그래프
- **Nodes:** 개별 작업 단위 (LLM 호출, 도구 실행 등)
- **Edges:** 노드 간 흐름 제어 (조건부 분기 포함)
- **Human-in-the-Loop:** 인간 개입 지점 설정

#### 학습 자료

- 🔗 [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- 🔗 [LangChain Academy](https://academy.langchain.com)

### 10.3 통합 프로젝트 아이디어

| 프로젝트 | 파인튜닝 요소 | 에이전트 요소 |
|----------|---------------|---------------|
| **도메인 전문가 에이전트** | 도메인 지식 파인튜닝 | RAG + 도구 사용 |
| **코드 리뷰 에이전트** | 코드 분석 능력 파인튜닝 | 파일 탐색, Git 연동 |
| **데이터 분석 에이전트** | SQL/분석 언어 파인튜닝 | 데이터베이스 연결, 시각화 |

---

## 전체 타임라인 요약

| 기간 | Phase | 핵심 산출물 |
|------|-------|-------------|
| 1-6주 | Phase 1: 기초 다지기 | nanoGPT 구현, Transformer 이해 |
| 7-9주 | Phase 2: HuggingFace 입문 | 모델 로드/추론, Tokenizer 이해 |
| 10-12주 | Phase 3: 데이터 준비 | 데이터 정제 파이프라인, 합성 데이터 |
| 13-18주 | Phase 4: LoRA/QLoRA 파인튜닝 | Text-to-SQL 모델, 한국어 도메인 모델 |
| 19-22주 | Phase 5: 트러블슈팅 | 디버깅 역량, 안정적 학습 |
| 23-26주 | Phase 6: DPO 정렬 | 선호도 정렬된 챗봇 모델 |
| 27-30주 | Phase 7: 평가 | 자체 평가 파이프라인, 벤치마크 결과 |
| 31-35주 | Phase 8: 배포 | vLLM 서버, Ollama 로컬 배포 |
| 36-40주 | Phase 9-10: 심화 (선택) | 최신 기법, 에이전트 통합 |

**총 예상 기간: 8-10개월** (주 10-15시간 투자 기준, 핵심 Phase 1-8)

---

## 필수 리소스 모음

### 무료 강좌

- [ ] Andrej Karpathy - Neural Networks: Zero to Hero (YouTube)
- [ ] Hugging Face NLP Course (huggingface.co/learn)
- [ ] fast.ai - Practical Deep Learning for Coders
- [ ] DeepLearning.AI - Generative AI with LLMs (Coursera)
- [ ] LangChain Academy - Free LangGraph Course

### 핵심 논문

- [ ] Attention Is All You Need (Transformer 원본)
- [ ] LoRA: Low-Rank Adaptation of Large Language Models
- [ ] QLoRA: Efficient Finetuning of Quantized LLMs
- [ ] Direct Preference Optimization (DPO)
- [ ] Scaling Laws for Neural Language Models

### 커뮤니티

- Hugging Face Discord/Forums
- r/LocalLLaMA (Reddit)
- Unsloth Discord
- AI Korea 커뮤니티

---

## 즉시 시작하기

### Week 1 Action Items

```
□ Karpathy의 "Let's Build GPT from Scratch" 2시간 영상 시청
□ nanoGPT 레포지토리 클론 및 환경 설정
□ GPU 환경 확보 (Colab/Kaggle 또는 로컬)
□ Hugging Face 계정 생성 및 토큰 발급
```

### 첫 파인튜닝까지

```
□ HuggingFace NLP Course Chapter 1-3 완료
□ Unsloth 설치 및 예제 노트북 실행
□ Alpaca 데이터셋으로 Llama 3.1 8B QLoRA 파인튜닝
□ 결과 모델 Hugging Face Hub에 업로드
□ GGUF 변환 후 Ollama로 로컬 테스트
```

---

## 실패 케이스 스터디

학습 과정에서 피해야 할 함정들:

### ❌ 케이스 1: 벤치마크만 높은 모델

```
문제: MMLU 점수는 높은데 실제 사용 시 답변 품질이 낮음
원인: 벤치마크 데이터가 학습 데이터에 오염됨
교훈: 데이터 오염 체크 필수, 자체 평가셋 구축
```

### ❌ 케이스 2: 과적합된 모델

```
문제: 학습 데이터와 비슷한 질문에만 잘 답함
원인: 데이터 다양성 부족, epoch 과다
교훈: 다양한 데이터 사용, early stopping, validation set 분리
```

### ❌ 케이스 3: 망각된 모델

```
문제: 파인튜닝 후 일반 상식이나 기존 능력이 사라짐
원인: 학습률 너무 높음, epoch 너무 많음
교훈: 낮은 학습률(1e-5), 소량 학습, 일반 데이터 혼합
```

### ❌ 케이스 4: 배포 불가 모델

```
문제: 학습은 됐는데 추론이 너무 느림/메모리 부족
원인: 양자화/최적화 고려 없이 학습
교훈: 배포 환경 먼저 정의, 양자화 테스트 선행
```

---

## 커뮤니티 참여 마일스톤

```
□ Hugging Face에 첫 모델 업로드
□ Model Card 작성 (학습 방법, 평가 결과 포함)
□ Reddit/Discord에 학습 결과 공유 및 피드백 받기
□ 다른 사람의 모델 평가하고 피드백 작성
□ 블로그/노션에 학습 여정 기록
□ 오픈소스 프로젝트에 기여 (이슈, PR)
```

---

> 💡 **Tip:** 각 Phase를 완료할 때마다 GitHub에 프로젝트를 공개하고, Hugging Face에 모델을 업로드하여 포트폴리오를 구축하세요. 실패한 실험도 기록해두면 나중에 큰 자산이 됩니다.
