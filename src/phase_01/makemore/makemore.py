"""
=============================================================================
makemore.py - 문자 수준 언어 모델 (Character-level Language Model)
=============================================================================

📺 강의: Andrej Karpathy - "Neural Networks: Zero to Hero" 시리즈
🔗 Makemore 시리즈:
   - Part 1: Bigram (https://youtu.be/PaCmpygFfXo)
   - Part 2: MLP (https://youtu.be/TCH_1BHY58I)
   - Part 3: BatchNorm (https://youtu.be/P6sfmUTpUmc)
   - Part 4: BackProp (https://youtu.be/q8SA3rM6ckI)
   - Part 5: WaveNet (https://youtu.be/t3YJ5hKiMQ0)
   - GPT from scratch (https://youtu.be/kCc8FmEb1nY)

📌 개요:
이 스크립트는 텍스트 데이터(한 줄에 하나씩)를 받아서 비슷한 것을 더 생성합니다.
예: 이름 데이터셋 → 새로운 그럴듯한 이름 생성

💡 지원하는 모델 아키텍처:
- Bigram: 가장 단순한 모델 (이전 1개 문자만 참조)
- MLP: Bengio et al. 2003 논문 구현
- RNN/GRU: 순환 신경망
- BoW: Bag of Words (Transformer의 단순화 버전)
- Transformer: GPT-2 스타일 (Self-Attention)

🔧 minGPT 대비 변경사항:
- GPT2 사전학습 가중치 로드 기능 제거
- Dropout 제거 (작은 모델에서 불필요)
- Weight Decay 복잡한 로직 제거 (이 규모에서 큰 차이 없음)

=============================================================================
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# 모델 설정 (Configuration)
# =============================================================================

@dataclass
class ModelConfig:
    """
    모델 하이퍼파라미터 설정 클래스

    📺 강의: GPT from scratch 약 00:15:00

    💡 @dataclass란?
    Python 3.7+에서 제공하는 데코레이터로,
    __init__, __repr__ 등을 자동 생성해줌

    Attributes:
        block_size: 입력 시퀀스 최대 길이 (컨텍스트 윈도우)
        vocab_size: 어휘 크기 (가능한 토큰 수)
        n_layer: Transformer 블록 개수 (깊이)
        n_embd: 임베딩 차원 (모델 너비)
        n_embd2: 일부 모델에서 사용하는 두 번째 임베딩 차원
        n_head: Attention 헤드 개수
    """
    block_size: int = None  # 입력 시퀀스 최대 길이
    vocab_size: int = None  # 어휘 크기 (문자 종류 수)
    # 아래 파라미터들은 모델 크기 조절용
    n_layer: int = 4        # Transformer 블록 수
    n_embd: int = 64        # 임베딩 차원
    n_embd2: int = 64       # 은닉층 차원 (MLP, RNN 등에서 사용)
    n_head: int = 4         # Attention 헤드 수


# =============================================================================
# Transformer 언어 모델 (GPT-2 스타일)
# =============================================================================
"""
📺 강의: "Let's build GPT: from scratch, in code, spelled out"
🔗 https://youtu.be/kCc8FmEb1nY

Transformer 아키텍처 구조:

    입력 토큰 (idx)
         ↓
    ┌─────────────────┐
    │  Token Embedding │  (vocab_size → n_embd)
    └─────────────────┘
         +
    ┌─────────────────┐
    │Position Embedding│  (block_size → n_embd)
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │  Transformer    │ ← n_layer번 반복
    │     Block       │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │   LayerNorm     │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │    LM Head      │  (n_embd → vocab_size)
    └─────────────────┘
         ↓
       logits
"""


class NewGELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) 활성화 함수

    📺 강의: GPT from scratch 약 00:45:00

    🧮 수식:
    GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    💡 왜 GELU?
    - ReLU보다 부드러운 비선형성
    - GPT, BERT 등 현대 Transformer에서 표준으로 사용
    - 음수 입력도 완전히 0이 되지 않음 (Dying ReLU 문제 없음)

    📊 비교:
    - ReLU: max(0, x) → 음수에서 완전히 0
    - GELU: 음수에서도 작은 값 통과 (확률적으로)

    🔗 논문: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    ==========================================================================
    Causal Self-Attention (인과적 자기 주의)
    ==========================================================================

    📺 강의: GPT from scratch 약 00:25:00 - 01:00:00 (핵심!)

    🎯 핵심 아이디어:
    "각 토큰이 자신 이전의 토큰들만 참조할 수 있도록" 제한된 Attention

    🧮 수학적 표현:
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    📊 예시 (시퀀스 "hello"):
    h → [h만 참조 가능]
    e → [h, e 참조 가능]
    l → [h, e, l 참조 가능]
    l → [h, e, l, l 참조 가능]
    o → [h, e, l, l, o 참조 가능]

    💡 왜 "Causal" (인과적)?
    - 미래 정보 누수 방지 (학습 시 정답을 미리 보면 안 됨)
    - 자기회귀(autoregressive) 생성에 필수
    - 삼각 마스크로 구현

    🔗 PyTorch 내장 MultiheadAttention 사용 가능하지만,
    여기서는 교육 목적으로 명시적 구현
    """

    def __init__(self, config):
        super().__init__()
        # n_embd가 n_head로 나누어 떨어져야 함
        assert config.n_embd % config.n_head == 0

        # ---- 핵심 레이어들 ----
        # Q, K, V를 한 번에 계산하는 선형 변환
        # 입력: (B, T, n_embd) → 출력: (B, T, 3*n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Attention 출력을 다시 n_embd 차원으로 변환
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # ---- Causal Mask (인과적 마스크) ----
        # 하삼각 행렬: 미래 토큰을 볼 수 없도록 마스킹
        #
        # 예: block_size=4일 때
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        #
        # 1 = 참조 가능, 0 = 참조 불가 (마스킹)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        순전파

        📺 강의: GPT from scratch 약 00:35:00

        Args:
            x: 입력 텐서 (B, T, C)
               B = batch size
               T = sequence length (시간 축)
               C = embedding dimension (n_embd)

        Returns:
            y: Attention 출력 (B, T, C)
        """
        B, T, C = x.size()  # batch, time(시퀀스), channels(임베딩)

        # --------------------------------------------------------------------
        # Step 1: Q, K, V 계산
        # --------------------------------------------------------------------
        # c_attn으로 한 번에 계산 후 3등분
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Multi-head로 분리: (B, T, C) → (B, nh, T, hs)
        # nh = number of heads, hs = head size = C // nh
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # --------------------------------------------------------------------
        # Step 2: Attention Score 계산
        # --------------------------------------------------------------------
        # QK^T / √d_k
        # (B, nh, T, hs) @ (B, nh, hs, T) → (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Causal masking: 미래 위치를 -inf로 설정
        # softmax(-inf) = 0 이므로 미래 토큰의 가중치가 0이 됨
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax로 확률 분포 변환
        att = F.softmax(att, dim=-1)

        # --------------------------------------------------------------------
        # Step 3: Value와 가중 합산
        # --------------------------------------------------------------------
        # (B, nh, T, T) @ (B, nh, T, hs) → (B, nh, T, hs)
        y = att @ v

        # Multi-head 결과 합치기: (B, nh, T, hs) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # --------------------------------------------------------------------
        # Step 4: 출력 projection
        # --------------------------------------------------------------------
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """
    Transformer Block: Attention + MLP + Residual Connection

    📺 강의: GPT from scratch 약 01:05:00

    구조 (Pre-LayerNorm 방식):

        x ─────────────────────────────────────→ (+) ─→ (+) ─→ output
            ↓                                     ↑       ↑
        ┌───────────┐   ┌─────────────────┐      │       │
        │ LayerNorm │ → │ Self-Attention  │ ─────┘       │
        └───────────┘   └─────────────────┘              │
                                                         │
        ┌───────────┐   ┌─────────────────┐              │
        │ LayerNorm │ → │      MLP        │ ─────────────┘
        └───────────┘   └─────────────────┘

    💡 왜 이 구조?
    - LayerNorm 먼저: 학습 안정성 향상 (Pre-LN)
    - Residual connection: gradient 흐름 개선, 깊은 네트워크 학습 가능
    - MLP: 비선형 변환으로 표현력 증가
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)     # Attention 전 정규화
        self.attn = CausalSelfAttention(config)     # Self-Attention
        self.ln_2 = nn.LayerNorm(config.n_embd)     # MLP 전 정규화

        # MLP: 확장 → 활성화 → 축소 (4배 확장이 표준)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),    # 4배 확장
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),   # 다시 축소
            act     = NewGELU(),                                      # 활성화 함수
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP 순전파 함수

    def forward(self, x):
        """
        순전파: Residual connection으로 연결

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
        """
        x = x + self.attn(self.ln_1(x))   # Attention + Residual
        x = x + self.mlpf(self.ln_2(x))   # MLP + Residual
        return x


class Transformer(nn.Module):
    """
    ==========================================================================
    Transformer 언어 모델 (GPT-2 스타일)
    ==========================================================================

    📺 강의: GPT from scratch 전체

    🎯 구조:
    1. Token Embedding: 각 토큰을 벡터로 변환
    2. Position Embedding: 위치 정보 추가
    3. Transformer Blocks: Self-Attention + MLP (n_layer번 반복)
    4. Final LayerNorm: 정규화
    5. LM Head: 다음 토큰 확률 분포 출력
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            # Token Embedding: 어휘 크기 → 임베딩 차원
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding: 위치 → 임베딩 차원
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Transformer Blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final LayerNorm
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Language Model Head: 임베딩 → 어휘 (다음 토큰 예측)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 파라미터 수 출력 (lm_head 제외, 보통 wte와 가중치 공유하므로)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        """
        순전파

        Args:
            idx: 입력 토큰 인덱스 (B, T)
            targets: 정답 토큰 (선택, 학습 시 사용)

        Returns:
            logits: 다음 토큰 로짓 (B, T, vocab_size)
            loss: Cross Entropy 손실 (targets 있을 때만)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"시퀀스 길이 {t}가 block_size {self.block_size}를 초과"

        # 위치 인덱스 생성: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # --------------------------------------------------------------------
        # 순전파
        # --------------------------------------------------------------------
        tok_emb = self.transformer.wte(idx)   # 토큰 임베딩 (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)   # 위치 임베딩 (1, t, n_embd)
        x = tok_emb + pos_emb                 # 둘을 더함 (브로드캐스팅)

        # Transformer 블록들 통과
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)          # 최종 LayerNorm
        logits = self.lm_head(x)              # 다음 토큰 예측

        # --------------------------------------------------------------------
        # 손실 계산 (학습 시)
        # --------------------------------------------------------------------
        loss = None
        if targets is not None:
            # Cross Entropy Loss
            # logits: (B*T, vocab_size), targets: (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)  # -1은 패딩, 무시

        return logits, loss


# =============================================================================
# Bag of Words (BoW) 언어 모델
# =============================================================================
"""
📺 강의: Makemore Part 5 (WaveNet 스타일)

BoW = Transformer의 단순화 버전
- Attention 가중치가 학습되지 않고 균등 평균
- "Transformer가 왜 좋은지" 이해하기 위한 베이스라인
"""


class CausalBoW(nn.Module):
    """
    Causal Bag of Words: 이전 토큰들의 단순 평균

    📺 강의: GPT from scratch 약 00:30:00

    💡 핵심:
    - Attention처럼 이전 토큰들을 집계하지만
    - 가중치가 학습되지 않고 균등 평균
    - Attention의 "단순화 버전"으로 이해

    📊 비교:
    Self-Attention: 학습된 가중치로 가중 평균
    BoW: 균등 가중치로 단순 평균
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # Causal mask: 미래 토큰 마스킹
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    def forward(self, x):
        """
        이전 토큰들의 평균 계산

        예: 위치 3에서
        - 위치 0, 1, 2, 3의 임베딩을 평균
        - (마스크로 인해 4 이후는 제외)
        """
        B, T, C = x.size()

        # 균등 가중치 Attention (학습 없음)
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # 행별로 정규화 → 균등 가중치

        # 가중 평균 (여기선 단순 평균)
        y = att @ x  # (B, T, T) @ (B, T, C) → (B, T, C)
        return y


class BoWBlock(nn.Module):
    """BoW + MLP 블록"""

    def __init__(self, config):
        super().__init__()
        self.cbow = CausalBoW(config)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    def forward(self, x):
        x = x + self.cbow(x)   # BoW + Residual
        x = x + self.mlpf(x)   # MLP + Residual
        return x


class BoW(nn.Module):
    """Bag of Words 언어 모델"""

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)     # 토큰 임베딩
        self.wpe = nn.Embedding(config.block_size, config.n_embd)     # 위치 임베딩
        self.context_block = BoWBlock(config)                          # BoW 블록
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)      # 출력층

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"시퀀스 길이 {t}가 block_size {self.block_size}를 초과"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.context_block(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss


# =============================================================================
# RNN 언어 모델 (Vanilla RNN / GRU)
# =============================================================================
"""
📺 강의: Makemore Part 4 (Backprop through time)

RNN = 순환 신경망
- 시퀀스를 순차적으로 처리
- 이전 상태(hidden state)를 다음 스텝에 전달
- Transformer 이전의 시퀀스 모델링 표준

GRU = Gated Recurrent Unit
- RNN의 개선 버전
- Reset/Update 게이트로 정보 흐름 제어
- LSTM보다 단순하지만 비슷한 성능
"""


class RNNCell(nn.Module):
    """
    Vanilla RNN Cell

    📺 강의: Makemore Part 4 약 00:20:00

    🧮 수식:
    h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)

    여기선 단순화하여:
    h_t = tanh(W · [x_t, h_{t-1}])

    💡 역할:
    현재 입력 x_t와 이전 상태 h_{t-1}을 받아
    새로운 상태 h_t를 출력
    """

    def __init__(self, config):
        super().__init__()
        # 입력(n_embd) + 이전상태(n_embd2) → 새 상태(n_embd2)
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        """
        Args:
            xt: 현재 입력 (B, n_embd)
            hprev: 이전 hidden state (B, n_embd2)
        Returns:
            ht: 새 hidden state (B, n_embd2)
        """
        xh = torch.cat([xt, hprev], dim=1)   # 입력과 이전 상태 연결
        ht = F.tanh(self.xh_to_h(xh))        # 선형 변환 + tanh
        return ht


class GRUCell(nn.Module):
    """
    GRU (Gated Recurrent Unit) Cell

    📺 강의: Makemore Part 4 약 00:35:00

    🧮 수식:
    z = σ(W_z · [x, h])           # Update gate
    r = σ(W_r · [x, h])           # Reset gate
    h̃ = tanh(W · [x, r * h])     # Candidate state
    h_new = (1-z) * h + z * h̃    # Final state

    💡 게이트 역할:
    - Reset gate (r): 이전 정보를 얼마나 "잊을지"
    - Update gate (z): 새 정보를 얼마나 "받아들일지"
    """

    def __init__(self, config):
        super().__init__()
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)     # Update gate
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)     # Reset gate
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)  # Candidate

    def forward(self, xt, hprev):
        # Reset gate: 이전 상태의 어떤 부분을 리셋할지
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev  # 리셋된 이전 상태

        # Candidate state: 새로운 후보 상태
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))

        # Update gate: 이전 상태와 후보를 어떻게 섞을지
        z = F.sigmoid(self.xh_to_z(xh))

        # 최종 상태: 이전과 새 상태의 가중 평균
        ht = (1 - z) * hprev + z * hbar
        return ht


class RNN(nn.Module):
    """
    RNN/GRU 언어 모델

    📊 구조:
    입력 시퀀스 → 토큰 임베딩 → [RNN Cell 반복] → 출력층 → logits
    """

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # 초기 hidden state (학습 가능)
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))

        # 토큰 임베딩
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # RNN Cell 선택
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)

        # 출력층
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # 모든 토큰 임베딩 (한 번에)
        emb = self.wte(idx)  # (B, T, n_embd)

        # 순차적으로 RNN 실행
        hprev = self.start.expand((b, -1))  # 초기 상태 (B, n_embd2)
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]               # i번째 토큰 임베딩 (B, n_embd)
            ht = self.cell(xt, hprev)       # RNN 스텝 (B, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # 모든 hidden state를 모아서 출력 계산
        hidden = torch.stack(hiddens, 1)    # (B, T, n_embd2)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss


# =============================================================================
# MLP 언어 모델 (Bengio 2003)
# =============================================================================
"""
📺 강의: Makemore Part 2 (MLP)
🔗 논문: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

역사적으로 중요한 모델:
- 신경망 언어 모델의 시초
- 단어 임베딩 개념 도입
- 이후 Word2Vec, GloVe 등에 영향
"""


class MLP(nn.Module):
    """
    MLP 언어 모델 (Bengio et al. 2003)

    📺 강의: Makemore Part 2 약 00:15:00

    🎯 핵심 아이디어:
    1. 이전 n개 토큰의 임베딩을 연결(concatenate)
    2. MLP로 다음 토큰 예측

    📊 구조:
    [이전 토큰들의 임베딩 연결] → MLP → 다음 토큰 확률

    예: block_size=3일 때
    [emb(t-3), emb(t-2), emb(t-1)] → MLP → P(t)
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # +1: 시퀀스 시작 전을 나타내는 <BLANK> 토큰용
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # MLP: (block_size * n_embd) → n_embd2 → vocab_size
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # 이전 block_size개 토큰의 임베딩 수집
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)  # 오른쪽으로 시프트
            idx[:, 0] = self.vocab_size  # 첫 위치는 <BLANK>
            embs.append(tok_emb)

        # 모든 임베딩 연결
        x = torch.cat(embs, -1)  # (B, T, n_embd * block_size)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss


# =============================================================================
# Bigram 언어 모델
# =============================================================================
"""
📺 강의: Makemore Part 1
가장 단순한 언어 모델: 직전 1개 문자만 보고 다음 문자 예측
"""


class Bigram(nn.Module):
    """
    Bigram 언어 모델

    📺 강의: Makemore Part 1 전체

    🎯 핵심:
    - 가장 단순한 언어 모델
    - 직전 1개 문자만 보고 다음 문자 예측
    - 단순 룩업 테이블 (신경망이라 부르기 민망할 정도)

    📊 구조:
    이전 문자 인덱스 → logits[이전문자] → 다음 문자 확률

    💡 학습:
    이 룩업 테이블이 학습되면서
    "a 다음에는 보통 뭐가 오는지"를 배움
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        # (vocab_size, vocab_size) 룩업 테이블
        # logits[i] = i번째 문자 다음에 올 각 문자의 점수
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1  # 이전 1개 문자만 필요

    def forward(self, idx, targets=None):
        # "순전파"라기보단 그냥 인덱싱
        logits = self.logits[idx]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss


# =============================================================================
# 유틸리티 함수들
# =============================================================================

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    모델에서 텍스트 생성

    📺 강의: GPT from scratch 약 01:40:00

    Args:
        model: 언어 모델
        idx: 시작 토큰 시퀀스 (B, T)
        max_new_tokens: 생성할 토큰 수
        temperature: 샘플링 온도 (높을수록 다양, 낮을수록 결정적)
        do_sample: True면 샘플링, False면 greedy (가장 확률 높은 토큰)
        top_k: 상위 k개 토큰에서만 샘플링 (선택)

    Returns:
        idx: 생성된 시퀀스 (B, T + max_new_tokens)

    💡 Temperature 효과:
    - temp = 1.0: 원래 확률대로
    - temp < 1.0: 확률 분포가 뾰족해짐 (더 결정적)
    - temp > 1.0: 확률 분포가 평평해짐 (더 다양)
    """
    block_size = model.get_block_size()

    for _ in range(max_new_tokens):
        # 시퀀스가 block_size보다 길면 자르기
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # 모델 순전파
        logits, _ = model(idx_cond)

        # 마지막 위치의 logits만 사용
        logits = logits[:, -1, :] / temperature

        # Top-k 필터링 (선택)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Softmax → 확률
        probs = F.softmax(logits, dim=-1)

        # 샘플링 또는 greedy 선택
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # 시퀀스에 추가
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def print_samples(num=10):
    """
    모델에서 샘플 생성 후 출력

    생성된 샘플을 세 카테고리로 분류:
    - train에 있음: 학습 데이터에 있는 단어
    - test에 있음: 테스트 데이터에 있는 단어
    - new: 새로 생성된 단어 (원하는 결과!)
    """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1  # -1: 시작 토큰 제외

    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')

    train_samples, test_samples, new_samples = [], [], []

    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()  # 시작 토큰 제외
        # 종료 토큰(0)에서 자르기
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)

        # 카테고리 분류
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)

    # 결과 출력
    print('-' * 80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-' * 80)


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    """
    모델 평가: 데이터셋에서 평균 손실 계산

    Args:
        model: 평가할 모델
        dataset: 평가 데이터셋
        batch_size: 배치 크기
        max_batches: 최대 배치 수 (전체 평가 시 None)

    Returns:
        mean_loss: 평균 손실
    """
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []

    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break

    mean_loss = torch.tensor(losses).mean().item()
    model.train()  # 다시 학습 모드로
    return mean_loss


# =============================================================================
# 데이터셋 클래스
# =============================================================================

class CharDataset(Dataset):
    """
    문자 수준 데이터셋

    📺 강의: Makemore Part 1 약 00:10:00

    각 단어를 (입력, 정답) 쌍으로 변환:
    - 입력: [<START>, 문자들...]
    - 정답: [문자들..., <END>]

    예: "hello"
    - x: [0, h, e, l, l, o, 0, 0, ...]  (0=시작/끝 토큰)
    - y: [h, e, l, l, o, 0, -1, -1, ...]  (-1=무시)
    """

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length

        # 문자 ↔ 인덱스 매핑 (0은 특수 토큰용으로 예약)
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        """단어가 데이터셋에 있는지 확인"""
        return word in self.words

    def get_vocab_size(self):
        """어휘 크기 (문자 종류 + 특수 토큰)"""
        return len(self.chars) + 1

    def get_output_length(self):
        """출력 시퀀스 최대 길이"""
        return self.max_word_length + 1  # +1: 시작 토큰

    def encode(self, word):
        """단어 → 인덱스 시퀀스"""
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ix):
        """인덱스 시퀀스 → 단어"""
        return ''.join(self.itos[i] for i in ix)

    def __getitem__(self, idx):
        """
        데이터 로드

        Returns:
            x: 입력 시퀀스 (시작 토큰 + 문자들)
            y: 정답 시퀀스 (문자들 + 끝 토큰)
        """
        word = self.words[idx]
        ix = self.encode(word)

        # 고정 길이 텐서 생성
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)

        x[1:1+len(ix)] = ix      # 입력: 인덱스 1부터 단어 시작
        y[:len(ix)] = ix         # 정답: 인덱스 0부터 단어 시작
        y[len(ix)+1:] = -1       # 패딩 위치는 -1 (손실 계산에서 무시)

        return x, y


def create_datasets(input_file):
    """
    입력 파일에서 학습/테스트 데이터셋 생성

    📺 강의: Makemore Part 1 약 00:05:00
    """
    # 파일 읽기
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]  # 공백 제거
    words = [w for w in words if w]     # 빈 문자열 제거

    # 어휘 및 통계
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)

    print(f"데이터셋 예제 수: {len(words)}")
    print(f"최대 단어 길이: {max_word_length}")
    print(f"어휘 크기: {len(chars)}")
    print(f"어휘: {''.join(chars)}")

    # 학습/테스트 분할 (90/10)
    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]

    print(f"학습 데이터: {len(train_words)}개, 테스트 데이터: {len(test_words)}개")

    # Dataset 객체 생성
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


class InfiniteDataLoader:
    """
    무한 반복 DataLoader

    💡 왜 필요한가?
    - 일반 DataLoader는 에폭 끝나면 StopIteration
    - 학습 루프에서 계속 배치를 받고 싶을 때 사용
    """

    def __init__(self, dataset, **kwargs):
        # replacement=True로 무한 샘플링
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # 명령줄 인자 파싱
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Makemore: 문자 수준 언어 모델")

    # 시스템/입출력 설정
    parser.add_argument('--input-file', '-i', type=str, default='names.txt',
                        help="입력 파일 (한 줄에 하나씩)")
    parser.add_argument('--work-dir', '-o', type=str, default='out',
                        help="출력 디렉토리")
    parser.add_argument('--resume', action='store_true',
                        help="기존 모델에서 이어서 학습")
    parser.add_argument('--sample-only', action='store_true',
                        help="학습 없이 샘플링만 수행")
    parser.add_argument('--num-workers', '-n', type=int, default=4,
                        help="데이터 로더 워커 수")
    parser.add_argument('--max-steps', type=int, default=-1,
                        help="최대 학습 스텝 (-1이면 무한)")
    parser.add_argument('--device', type=str, default='cpu',
                        help="연산 장치 (cpu, cuda, mps 등)")
    parser.add_argument('--seed', type=int, default=3407,
                        help="랜덤 시드")

    # 샘플링 설정
    parser.add_argument('--top-k', type=int, default=-1,
                        help="Top-k 샘플링 (-1이면 비활성화)")

    # 모델 설정
    parser.add_argument('--type', type=str, default='transformer',
                        help="모델 종류: bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4,
                        help="레이어 수")
    parser.add_argument('--n-head', type=int, default=4,
                        help="Attention 헤드 수 (Transformer)")
    parser.add_argument('--n-embd', type=int, default=64,
                        help="임베딩 차원")
    parser.add_argument('--n-embd2', type=int, default=64,
                        help="은닉층 차원")

    # 최적화 설정
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help="배치 크기")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4,
                        help="학습률")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01,
                        help="가중치 감쇠")

    args = parser.parse_args()
    print(vars(args))

    # -------------------------------------------------------------------------
    # 시스템 초기화
    # -------------------------------------------------------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # -------------------------------------------------------------------------
    # 데이터셋 준비
    # -------------------------------------------------------------------------
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"데이터셋에서 결정: {vocab_size=}, {block_size=}")

    # -------------------------------------------------------------------------
    # 모델 초기화
    # -------------------------------------------------------------------------
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2
    )

    # 모델 타입에 따라 생성
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f"알 수 없는 모델 타입: {args.type}")

    model.to(args.device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")

    # 기존 모델 로드 (resume 또는 sample-only 시)
    if args.resume or args.sample_only:
        print("기존 모델 로드 중...")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))

    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # -------------------------------------------------------------------------
    # 옵티마이저 초기화
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8
    )

    # -------------------------------------------------------------------------
    # 데이터 로더 초기화
    # -------------------------------------------------------------------------
    batch_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # -------------------------------------------------------------------------
    # 학습 루프
    # -------------------------------------------------------------------------
    """
    📺 강의: Makemore 시리즈 전체

    학습 루프 구조:
    1. 배치 로드
    2. 순전파 (forward)
    3. 손실 계산
    4. 역전파 (backward)
    5. 파라미터 업데이트
    6. 로깅 및 평가
    """
    best_loss = None
    step = 0

    while True:
        t0 = time.time()

        # 배치 로드 및 디바이스 이동
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # 순전파 및 손실 계산
        logits, loss = model(X, Y)

        # 역전파 및 파라미터 업데이트
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # CUDA 동기화 (시간 측정용)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # 로깅 (10 스텝마다)
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # 평가 (500 스텝마다)
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")

            # 최고 성능 모델 저장
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss}가 최고 성능! 모델 저장: {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # 샘플 생성 (200 스텝마다)
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1

        # 종료 조건
        if args.max_steps >= 0 and step >= args.max_steps:
            break
