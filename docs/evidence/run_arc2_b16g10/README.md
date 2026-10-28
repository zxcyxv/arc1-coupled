# ARC-2 16/10 학습 2D RoPE 런 (`runs/arc2_b16g10`, 10-26, 중단)

- **무엇을 했나:** ARC-2 TRM 분할(training2 1000 + evaluation2 120 시범 + ConceptARC 160), 16/10, 학습 2D RoPE, 행렬 wd 0.1,
  λ 학습, GA 4, 10k 에폭 계획. ARC-1 16/10 런과 같은 설정.
- **결과:** 1k 에폭 quick 평가에서 원본 판 0/172, view 0, 칸 정확도 0.749. 7,440스텝(약 1.03k 에폭)에서 체크포인트를 쓰고
  멈췄다(ARC-1 wd 1.0 런으로 GPU를 넘김). 마지막 학습 지표: loss 0.35, train exact 0.13, 칸 0.875.
- 이후 ARC-2는 1D 고정 RoPE와 λ 0.005 고정으로 다시 돌렸다(`run_arc2_b16g10_rope1d`, 1만 에폭까지 모든 평가 0/172).
