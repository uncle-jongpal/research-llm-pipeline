# DPO 파인튜닝 실습

> SFT 모델에 "판단력"을 심는 실전 — TRL DPOTrainer로 선호도 학습을 실행한다

---

## 1. DPO 학습 전체 흐름

```mermaid
flowchart TD
    A["Phase 4 SFT 모델"] --> B["QLoRA 로드<br/>+ LoRA r=16"]
    B --> C["UltraFeedback 1000개<br/>prompt+chosen+rejected"]
    C --> D["DPOTrainer<br/>β=0.1, lr=5e-5, 1 epoch"]
    D --> E["DPO 메트릭 분석<br/>margins, accuracies"]
    E --> F["Before/After 비교"]
    F --> G["어댑터 저장"]
```

---

## 2. DPOTrainer 핵심 설정

| 인자 | 값 | SFT 대비 차이 | 이유 |
|------|---|-------------|------|
| learning_rate | 5e-5 | SFT 2e-4의 1/4 | reference에서 벗어나지 않도록 |
| num_train_epochs | 1 | SFT 3의 1/3 | 과도한 정렬 방지 |
| beta | 0.1 | SFT에 없음 | KL divergence 제약 강도 |
| per_device_batch | 1 | SFT 2 | chosen+rejected 쌍으로 메모리 2배 |
| ref_model | None | - | 자동으로 policy 복사본 사용 |

---

## 3. DPO 메트릭 해석

```mermaid
flowchart TD
    A["DPO 학습 중"] --> B["rewards/margins<br/>chosen - rejected 차이"]
    A --> C["rewards/accuracies<br/>올바른 선호 구분 비율"]
    A --> D["loss<br/>DPO 손실"]

    B --> E{"> 0?"}
    E -->|YES| F["정상: chosen이 더 높은 reward"]
    E -->|NO| G["문제: 선호도 역전"]

    C --> H{"> 0.5?"}
    H -->|YES| I["정상: 과반수 올바르게 구분"]
    H -->|NO| J["문제: 데이터 또는 학습 실패"]
```

| 메트릭 | 의미 | 건강한 값 |
|--------|------|----------|
| rewards/margins | chosen-rejected reward 차이 | > 0, 증가 추세 |
| rewards/accuracies | 올바른 선호 구분 비율 | > 0.5, 이상적으로 > 0.7 |
| loss | DPO 손실 | 감소 추세 |

---

## 4. SFT vs DPO 비교

| 단계 | SFT (Phase 4) | DPO (Phase 6) |
|------|-------------|--------------|
| 입력 데이터 | instruction + output | prompt + chosen + rejected |
| 학습 목표 | 정답을 따라해라 | 좋은 답을 선호해라 |
| lr | 2e-4 | 5e-5 (더 보수적) |
| epoch | 3 | 1 (보통) |
| 배치 메모리 | 1x | 2x (chosen+rejected 쌍) |
| 핵심 HP | r, alpha | β, lr |
| 평가 메트릭 | eval_loss | rewards/margins, accuracies |

---

## 5. 어댑터 스택

```mermaid
flowchart LR
    A["Qwen 2.5 1.5B<br/>(원본, 동결)"] --> B["SFT 어댑터<br/>→ 지시 따르기"]
    A --> C["DPO 어댑터<br/>→ 응답 품질 향상"]

    B -.->|"SFT 위에 DPO"| C
```

DPO는 SFT 위에 쌓는 구조. SFT가 "무엇을 답할지"를 배우고, DPO가 "어떻게 더 잘 답할지"를 배운다.

---

## 핵심 원칙

> **DPO는 SFT 위에 쌓는 "2층 건물"이다. 1층(SFT)이 부실하면 2층(DPO)도 무너진다.**
> SFT의 품질이 DPO의 상한선을 결정한다.
