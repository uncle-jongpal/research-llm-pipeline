# LLM Enhance — LLM 파인튜닝 & 배포 학습

LLM 파인튜닝부터 배포까지 전 과정을 실습하는 학습 프로젝트입니다.

---

## 프로젝트 구조

```
00_llm_enhance/
├── src/                    # 실습 노트북
│   ├── phase_01/           # 기초 (nanoGPT, micrograd 등)
│   ├── phase_02/           # Attention, Transformer
│   ├── phase_03/           # 데이터 파이프라인, chat template
│   ├── phase_04/           # LoRA/QLoRA, SFT 파인튜닝
│   ├── phase_05/           # Catastrophic Forgetting, W&B
│   ├── phase_06/           # DPO, RLHF, GRPO
│   ├── phase_07/           # 벤치마크, 커스텀 평가셋, Human Eval
│   ├── phase_08/           # 양자화, vLLM, Ollama 배포
│   ├── phase_09/           # Mamba/SSM, 고급 학습 기법
│   └── phase_10/           # 에이전트, LangGraph
├── summary/                # Phase별 요약 문서
└── README.md
```

---

## 환경 설정

### 1. 가상환경 생성

```bash
conda create -n distillation python=3.10 -y
conda activate distillation
```

### 2. 패키지 설치

```bash
pip install torch transformers peft accelerate trl bitsandbytes datasets
pip install matplotlib pandas numpy jupyter
```

(각 Phase 노트북에서 추가 패키지가 필요하면 `!pip install`로 설치)

### 3. GPU (권장)

- CUDA 지원 GPU (RTX 4070 Ti 12GB 이상 권장)
- Phase 4 SFT, Phase 6 DPO 학습에 필요

---

## 실행 방법

1. Jupyter Lab 또는 VS Code에서 `src/` 내 노트북 실행
2. Phase 순서대로 진행 권장 (2 → 3 → 4 → …)
3. `output/` 폴더는 학습 결과물 저장용 (git 제외)

---

## 주요 Phase 요약

| Phase | 내용 |
|-------|------|
| 2 | Attention, Transformer 구조 |
| 3 | 데이터 형식, chat template, 합성 데이터 |
| 4 | LoRA/QLoRA, SFT 첫 파인튜닝, Text-to-SQL |
| 5 | Catastrophic Forgetting, W&B 실험 관리 |
| 6 | DPO, Preference 학습, GRPO |
| 7 | MMLU/GSM8K, 커스텀 평가셋, Human Eval |
| 8 | GPTQ/AWQ 양자화, vLLM, Ollama 배포 |
| 9 | Mamba/SSM, 고급 학습 기법 |
| 10 | 파인튜닝 모델 + 에이전트, LangGraph |

---

## 라이선스

학습용 프로젝트입니다.
