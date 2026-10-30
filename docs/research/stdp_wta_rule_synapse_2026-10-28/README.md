# 조사: 규칙 시냅스의 lasso를 STDP + WTA로 대체할 수 있는가 (2026-10-28)

> 성격: 10-28 저녁 사용자 요청("lasso의 역할을 STDP + WTA로 대체할 수 있는지, 아키텍처와 설명이 깔끔하게 떨어지는지")으로 돌린
> 다중 에이전트 조사(12개)의 기록이다. 아래 "최종 권고"가 종합 에이전트의 보고서 전문이고, 근거 스크립트와 각 에이전트의
> 원 결과도 이 폴더에 있다. 수치는 대부분 스크립트로 재계산된 것이지만, 인용 표시 [U]는 원문 확인이 필요하다.

## 진행 방식

1. 기반 조사 3개(병렬): 코드와 이전 설계 문서(theorem 0, 명제 4), 문헌(원문 확인 여부 표시), 이론(경쟁 동역학의 고정점).
2. 설계 4개(서로 다른 관점) → 각 설계마다 반박 검토 1개(이론가·엔지니어·심사위원 관점, 0~5점).
3. 종합 1개: 반박 점수를 기준으로 최종 권고.

| 관점 | 설계 | lasso 대체 | 우아함 | 이론 | 구현 | 검증 가능성 |
|---|---|---:|---:|---:|---:|---:|
| minimal-elegance | Rule synapse with a predict-then-observe STDP window and a sparsemax winner-take-all post. The constant depression with a floor (the soft threshold) is kept as the selection term. | 1.5 | 2 | 3 | 3.5 | 3 |
| guarantee-first | Competitive delta-rule synapse: an LCA rule synapse with a post-leading recursion window and E%-max competition, whose fixed point is the lasso | 2 | 2 | 2 | 3.5 | 2.5 |
| biology-faithful | Plateau-gated three-factor rule synapse with a soft winner-take-all colour readout | 1.5 | 2 | 2 | 3.5 | 2.5 |
| expressivity-for-ARC-2 | Free/clamped delta-rule heads under soft WTA: a TEM-style rule synapse (no lasso) | 3 | 2.5 | 2.5 | 3.5 | 3 |

## 파일

- `agent_results.json`: 에이전트 12개의 원 결과(기반 조사, 설계, 반박 검토, 종합).
- `scripts/`: 에이전트가 쓴 검사 스크립트와 출력. `theory/`(P1, 유한 λ, 원환 WTA 검사), `theory2/`(보장 우선 설계의 회로 검사),
  `theory_wta/`, `wta/`(경쟁 검사), `review_cdr/`, `review_wta/`, `review/`(반박 검토의 재계산: 배치 영역 스펙트럼, drift, 인증값).
  scratchpad에서 옮겼으므로 스크립트 안의 경로는 고쳐야 다시 돌아간다.

---

# Lasso를 STDP + WTA로 바꾸면 아키텍처와 설명이 깔끔해지는가: 최종 권고 (2026-10-28)

## 1. 결론

부분적으로만 그렇다. 설명은 깔끔해지고, 메커니즘 교체는 그렇지 않다.

STDP + WTA는 lasso를 대체하지 못한다. 근거는 네 가지이고, 모두 계산으로 확인했다.

- **선택 원리는 ℓ1이 맡는다.** torus 위치 연산자 검사(512개 인스턴스, 정답은 색 행마다 0이 아닌 항 1개)의 질의 정답률은 다음과 같다.
  - 수렴한 lasso 0.914
  - WTA 쓰기 + ℓ1 floor 0.914
  - floor 없는 WTA 0.006
  - 최소노름 최소제곱 0.012
- **hard WTA(k-WTA = IHT)는 보장을 약하게 만든다.** 잡음 없는 데모에서는 문턱보다 큰 항으로 된 exact fit이 모두 고정점이다. 그래서 선택이 시작점과 경로로 넘어간다. 보장 조건도 NSP(필요충분)에서 RIP_3s로 바뀐다. RIP_3s는 충분조건일 뿐이고 학습된 코드에서는 인증할 수 없다. 실측 회복률도 lasso보다 낮다: n=30에서 0.50 vs 1.00, 고정 Fourier 코드 top-1에서 9.1% vs 96.3%.
- **시간 비대칭 커널은 수렴점의 답을 바꾸지 못한다.** DC 이득이 양수이고 색마다 같으면, 수렴한 θ는 lasso 해의 양수배다(P1). 홀수 부분은 경로만 바꾼다.
- **어떤 변형도 가설족(조건 S)을 넓히지 않는다.** 따라서 ARC-2 0/172는 그대로다.

반면 설명은 코드를 한 줄도 바꾸지 않고 깔끔해진다. 지금 rule head의 한 걸음은 정확히 LCA(Rozell et al. 2008)의 전진 Euler 한 걸음이다. λ=0이면 DeltaNet 꼴의 델타 규칙이다. WTA로 얻으려던 경쟁은 이미 시냅스 사이의 볼록 경쟁(측방 억제 + 바닥 있는 균일 depression)으로 들어 있다. STDP가 일할 자리는 "정상 상태에서 사라지는 항은 보장을 못 바꾸고 경로만 바꾼다"는 원리 하나로 정해진다. 그 자리는 지각(bands), 수렴 전의 유한 블록, 비볼록 탐색(torus)이다.

**권고: 코드는 유지하고 설명을 다시 쓴다.** 네 교체 설계는 모두 부품을 늘렸고, 실제 배치 영역에서 측정된 이득이 없다.

### 1.1 네 제안의 판정 (비평 점수: 대체 / 우아함 / 이론, 5점 만점)

| 제안 | 점수 | 치명 결함 (비평이 재계산으로 확인) | 가져온 것 |
|---|---|---|---|
| minimal-elegance: sparsemax post + ℓ1 floor | 1.5 / 2 / 3 | ℓ1이 여전히 선택 원리다. 바뀌는 것은 데이터 항(보간 대신 margin)뿐이다. 8블록에서는 두 쓰기가 같다(0.770 vs 0.772). STDP 창은 한 블록 안의 가상 두 시점이고 학습 파라미터가 없다. 충분통계 G, S를 잃는다. | floor 실험(0.914 vs 0.006). λ→0+ 극한이 min-ℓ1 margin 규칙이라는 정리. |
| guarantee-first: LCA + 회귀 창 + E%-max | 2 / 2 / 2 | 창은 같은 시각 곱 e_t에 건 eligibility trace, 즉 heavy-ball momentum이지 pre/post 창이 아니다. Kbar가 양수라는 보장이 없다. 0.25 clamp는 고정점을 lasso(2.5λ)로 옮기고, Kbar<0이면 발산한다. 배치 영역은 고정점에서 멀다. | 고정점 위치 증명 기법(R_c ≤ ωR_c ⇒ R_c=0). 조직 문장 "정상 상태에서 무엇이 남는가". |
| biology-faithful: softmax WTA + 3요소 | 1.5 / 2 / 2 | "작은 신호 극한 = 지금 λ=0 단계"는 거짓이다. 계수가 2ρ와 2/K이고 상대오차가 0.82다. max-ent는 적합의 쌍대 기술일 뿐 θ를 고르지 않는다. 실제 선택자는 θ_0=0의 min-norm이다. 시간 비대칭이 아무것도 계산하지 않는다. | net LTD(ρ<1)와 유계성의 대응. torus의 critic 처방. |
| expressivity: free/clamped 쌍 + soft-WTA 헤드, λ 제거 | 3 / 2.5 / 2.5 | warm start에 φ drift가 겹치면 (I − ηG_t)가 null space 성분을 지우지 못한다. 그래서 min-norm 주장과 span 인증이 성립하지 않는다. "transition rows"는 같은 깊이의 행이라 transition이 아니다. 실험 1이 변화 네 가지를 묶어서 결과를 어느 변화에도 귀속할 수 없다. | 유한-T 항등식(이론의 중심축). ℓ1의 다섯째 기능(망각). 측정된 G 스펙트럼. 열린 문제로서의 지연 목표. |

E%-max 문턱은 고정 코드 toy에서 유한 걸음 가속이 실제로 있다(T=16: 기본값 76.6%, FISTA+E%-max 66.6%, 문턱을 최적화한 FISTA 57.7%). 다만 이 가속은 잔차 적응 문턱(AMP류) 덕분이지 STDP나 WTA 덕분이 아니다. ≤10 하네스에서는 수렴이 병목도 아니다(ISTA-30이 r-only 30–50k 평균을 60.3에서 64.8로, +4.5). 그래서 설계에 넣지 않는다.

## 2. 최종 설계 (forward 코드는 지금과 같다)

### 2.1 식

```
레인 l, 같은 그룹의 다른 데모 레인 l' ∈ others(l), 출력 칸 o, 블록 t
w_l'(o) = m_l'(o)·demo_l' ,   n_l = max(1, Σ_l' Σ_o w_l'(o))
φ_t(o) = unit(W_φ h_t(o)) ∈ R^64 ,  κ_t,k(o) ∈ R^13 (k=0..K, Σ_k|κ_k|² ≤ 1) ,  y(o) = onehot13(label)

G_t = (1/n_l) Σ_l' Σ_o w φ_t φ_tᵀ      [64×64]   논문 표기 Σ_φ
S_t = (1/n_l) Σ_l' Σ_o w y φ_tᵀ        [13×64]   논문 표기 C_yφ (bands의 S, 구조코드 G와 충돌하므로 개명)
r̂_t(o) = θ_t φ_t(o) + Σ_k Γ_k κ_t,k(o)     (레인 l의 시냅스가 l'의 칸을 예측한 값)
e_t  = S_t − θ_t G_t − Σ_k Γ_k C_t,k  =  (1/n_l) Σ_l' Σ_o w (y − r̂_t) φ_tᵀ
θ_{t+1} = soft(θ_t + e_t, λ);  이어서 Γ도 같은 꼴 (Gauss-Seidel, step 1)
r_{t+1}(o) = θ_{t+1} φ_t(o) + Σ_k Γ_{t+1,k} κ_t,k(o)  (자기 칸)  → W_fb → 다음 블록;  logits = classifier(h) + τ r
```

코드는 셀 꼴이 아니라 충분통계 S − θG로 계산한다. 두 식은 같다. 레인 사이 연산은 statistics()의 others@P 하나뿐이다. 그래서 명제 4가 코드 구조로 보장된다.

### 2.2 정확한 읽기 세 가지 (항등식, float64로 확인)

- **I1, LCA.** u_{t+1} = θ_t + e_t = S_t − θ_t(G_t − I)이고 θ_{t+1} = soft(u_{t+1}, λ)다. 이것은 Rozell의 u̇ = (1/τ)[b − u − (ΦᵀΦ − I)a], a = T_λ(u)를 dt = τ로 한 걸음 간 것이다. 대응은 b = S, 측방 억제 = G − I다. 13개 색 행이 G를 공유하는 LCA 13개다(최대 차 2.8e-14).
  - S는 교사 × pre로 된 Hebbian 구동이다.
  - θG는 자기 read × pre로 된 anti-Hebbian 항이다. 색 단위 c로 들어오는 시냅스 (c,j)와 (c,k)는 G_jk에 비례해 서로 억제한다. G_jk는 다른 데모 칸들에서 특징 j와 k가 함께 켜진 정도다. LISTA는 같은 항을 "mutual inhibition / explaining away"라 부른다.
  - soft는 0을 바닥으로 하는 일정 크기의 depression이다.
- **I2, 델타 규칙.** λ=0이면 θ_{t+1} = θ_t(I − G_t) + S_t다. 이것은 DeltaNet의 W(I − βkkᵀ) + βvkᵀ를 데모 칸 전체에 대해 합친 꼴이다(β = 1/n). 두 기억은 같은 계보 안에서 감쇠 연산자로 갈린다. bands는 스칼라·복소 감쇠((1−η), z_r)를 쓰는 더하기형 fast weight다. rule head는 데이터 의존 감쇠 (I − G_t)를 쓰고 근접 사상이 붙는다. 이것은 계보 진술이지 통합 법칙 주장이 아니다.
- **I3, Dale 분할.** θ = θ⁺ − θ⁻, θ^± ≥ 0으로 나누고 각 집단을 같은 델타 항, 균일 depression λ, 0에서의 정류로 갱신하면 고정점이 같다. 최적점에서는 θ⁺θ⁻ = 0이다(9.8e-15). 궤적은 다르므로 고정점 동치로만 말한다.
  - 활동이 부호 있는 집단 부호이므로 LTP/LTD는 연관을 기준으로 정의한다.
  - LTP-like는 연관 yφᵀ 쪽으로 움직이는 것이다.
  - LTD-like는 −r̂φᵀ 방향(예측한 연관에서 멀어지는 쪽)으로 움직이는 것, 또는 |θ|를 0 쪽으로 줄이는 균일 depression이다.

### 2.3 명제

**P1. 고정점 위치: 시간 커널 불변성. 증명됨, 수렴은 전제.**

- 설정: Hebbian 항(y⊗φ)과 anti-Hebbian 항(r̂⊗φ)에 각각 양면 커널을 건다. pre가 앞서는 로브 K⁺(τ)는 post_t ⊗ pre_{t−τ}, post가 앞서는 로브 K⁻(τ)는 post_{t−τ} ⊗ pre_t에 곱하고, 같은 시각 항 K⁰가 있다. 두 항의 DC 이득을 k_h, k_a라 한다.
- 전제:
  - 실제 DC 값이 k_h, k_a > 0이다. clamp를 쓰면 고정점이 움직인다.
  - 색마다 같고 특징별 이득이 없다.
  - (φ_t, θ_t)가 수렴한다.
- 결론: θ_∞ = (k_h/k_a)·lasso(G*, S*, λ/k_h).
- 증명: 상수 궤적에서는 시차 곱이 모두 같은 시각 곱과 같다. 그러면 k_h S* − k_a θG* ∈ λ∂‖θ‖₁이 되고, θ' = (k_a/k_h)θ로 바꾸면 lasso의 KKT 조건이다.
- 따름: 홀수 부분은 상수 궤적에서 정확히 0이다. λ→0+에서 질의 argmax가 lasso와 같으므로 theorem 0이 그대로 선다. float64 차 2.3e-13.
- 단서:
  - 색마다 DC가 다르면 argmax가 바뀐다(일치율 0.878). 색 순열 등변성이 이것을 막는다.
  - 특징별 DC d_j는 λ_j = λ/d_j인 가중 lasso를 만든다. 이것은 DC 효과이지 홀수 로브 효과가 아니다.
  - 수렴이 증명된 경우는 ISTA와 α-leaky 한쪽 trace(Krasnosel'skii-Mann, Balavoine et al. 2012)뿐이다.

**P2. 비대칭이 답을 바꿀 수 있는 곳(범위).** 네 곳뿐이다.
1. 수렴하지 않는 지각. 순환 궤도에서 평균 고정점은 단조 변분부등식을 푼다. 그 비대칭 부분 Σ K_odd(τ)(C(τ) − C(τ)ᵀ)는 궤도의 순환량이고, 이때 해는 어떤 목적함수도 최소화하지 않는다. 같은 DC에서 odd와 even 커널의 차가 노름의 약 3%다.
2. 유한 블록.
3. 비볼록 선택(hard, k-WTA, mixture).
4. (지각, θ) 결합 계가 어느 φ*에 도달하느냐.

**P3. 유한-T 항등식: 배치 영역의 이론 중심축. 증명됨.**

- 설정: φ 고정, λ=0, θ_0 = 0, 걸음 η.
- θ_T = S G⁺(I − (I − ηG)^T)이다. 이것은 Landweber 반복이다.
- 데모가 실현 가능하면(S = θ*G), 질의 오차는 정확히 r_T(o*) − θ*φ*(o*) = −θ*(I − ηG)^T φ*(o*)다(6.7e-15).
- 이 식 하나가 두 역할을 한다.
  - 잡음 필터: 고유값 s인 방향은 1 − (1 − ηs)^T만큼만 맞춘다.
  - 보장: T→∞에서 오차가 0이 될 조건은 φ*가 range(G) 안에 있는 것이다.
- 훈련된 ≤10 코드에서 잰 값:
  - λ_max(G)의 중앙값 0.31–0.61 (tr G = 1)
  - 유효 rank tr/λ_max 약 1.6–3.1
  - 질의 칸 |(I − G)^8 φ_q|의 중앙값 0.28 (η=1)
  - 1/λ_max는 지금 step 1의 1.6–3.1배다. 제안서의 "55배"는 등방 랜덤 코드에서 나온 값이라 틀렸다.
- 배치 영역은 고정점에서 멀다(b8g4_rope1d_step50000_ema에서 측정). φ가 블록마다 0.27–0.92 움직이고, θ의 KKT 잔차가 λ의 10–20배다. 그래서 P3는 배치 영역의 모형이지 정확한 기술이 아니다.

**Theorem 0 (옛 문서, 그대로).** S, L, I(NSP), C가 성립하면 lasso 최소해가 참 규칙이고 질의 답이 정확하다.

- 유한 λ 판: irrepresentable 조건과 sign consistency를 가정하면 θ_S = θ*_S − λG_SS⁻¹ sign(θ*_S)다. 질의 margin이 λ|φ_Sᵀ G_SS⁻¹ sign|보다 크면 argmax가 정확하다(1.25e-12). P1에 의해 이 결과는 DC 양수 커널 전부로 넘어간다(λ → λ/k_h).
- 논문에서는 "theorem 0은 8–16블록이 도달하지 않는 결합 고정점에 대한 진술"이라고 명시해야 한다.

### 2.4 lasso의 기능별로 무엇이 맡는가

| lasso 기능 | 최종 설계에서 | 근거와 상태 |
|---|---|---|
| 1. 선택(min ℓ1) | 그대로 soft threshold(두 Dale 집단의 균일 depression + floor)가 맡는다. 수렴하면 min ℓ1. 배치된 8–30블록에서 실제 선택자는 θ_0 = 0에서의 조기 정지(P3의 스펙트럼 필터, min-norm 편향)다. WTA로는 대체할 수 없다. | torus 0.914 vs floor 없음 0.006. IHT 0.50 vs 1.00. |
| 2. explaining-away(θG) | 시냅스 사이 측방 억제(LCA). 그대로. | I1 항등식 |
| 3. 잡음 문턱 λ≈κ/√n | 배치 영역에서는 유한-T 필터가 약한 고유방향을 누른다. λ의 보정값은 실제로 일하지 않는다. 원칙적 대안이 필요하면 항상성 문턱(목표 support 크기의 Lagrange 승수, SAILnet)이 있지만 넣지 않는다. | E1 15k(λ=0 / 0.01): direct 8/4 74/67, direct 16/10 84/87, r-only 8/4 45/47, r-only 16/10 50/47. 세 설정은 ±3 안이고 direct 8/4만 λ=0이 7 높다(grounding의 "모두 같다, 68"은 틀림). ARC-2 0.0025 vs 0.005는 차이 없음. 스펙트럼 필터라는 해석은 가설. |
| 4. 보장 | theorem 0 그대로. P1로 커널까지 확장. 유한-λ 닫힌 꼴. 배치 영역은 P3의 정확한 오차 식. | 전제 C는 배치에서 성립하지 않는다(KKT 10–20λ). |
| 5. 망각(원래 목록에 없음) | warm start에서 φ가 움직이면 (I − G_t)는 G_t의 null space 성분을 그대로 둔다. 묵은 성분을 지우는 것은 soft threshold뿐이다. 그래서 loops=16(128걸음)인 full ARC에서는 λ>0을 유지한다. | λ 없는 Landweber를 8블록×4걸음 drift시키면 \|Θ(I − P_final)\| = 12.4 (min-norm이면 0). |

### 2.5 증명된 것, 측정된 것, 추측

- **증명**(대수, float64 확인):
  - I1–I3
  - α-leaky LCA의 수렴
  - P1(수렴 전제)
  - P2(a)의 변분부등식
  - P3
  - hard threshold는 고정점에서 아무것도 고르지 않는다(한 줄 증명; 시작점 6개에서 서로 다른 dense 해 6개)
  - 유한-λ 닫힌 꼴(표준 결과, 출처는 기억이라 확인 필요)
- **측정**:
  - torus와 IHT 수치
  - 배치 영역 수치(φ drift, KKT, 스펙트럼, 인증값)
  - E1, ISTA-30
- **추측과 해석**:
  - E1의 λ 무효과가 스펙트럼 필터 때문이라는 설명
  - Sudoku에서 순 DC가 0 근처인 헤드 7/8은 "변화를 기록한다"는 읽기
  - TEM과의 연결(형태의 유사, 오차 인자만 일치)
  - 일반 커널의 수렴
  - 결합 고정점의 존재와 도달 가능성

## 3. 이야기 (한 문단)

Hebb의 가설은 인과적이었다("발화에 참여한다"). 형식화된 규칙 Δw ∝ post·pre는 시간을 버렸다. fast weight → linear attention(Schlag et al. 2021) → 델타 규칙(DeltaNet)의 계보가 그 무시간 형태를 물려받았다. HRM, TRM, URM 같은 재귀 추론기는 재귀 깊이라는 시간축을 만들었지만, 그 축을 가로지르는 기억은 두지 않는다. 이 모델은 그 축 위에 빠른 기억 두 개를 둔다.

쌍 안(지각)에서는 각 헤드의 bands 기억이 같은 토큰을 서로 다른 깊이에서 짝짓는다. 이것은 eligibility trace가 있는 시간 비대칭 Hebbian 규칙이다. 학습된 양면 커널의 DC 부분이 Hebbian 항이다. 이 DC는 부호가 있어서, Sudoku에서는 8개 중 7개 헤드가 고정점 순이득을 약 0으로 만든다. 홀수 부분은 궤적을 기록하고 고정점에서 사라진다(Xie & Seung 1999).

쌍 사이(데모에서 질의로 가는 유일한 길, 명제 4)에서는 rule head가 다른 데모들 위에서 푼 델타 규칙이다. 교사와는 Hebbian, 자기 예측과는 anti-Hebbian이다. 전시냅스 특징이 함께 켜지는 만큼 시냅스끼리 측방 억제하고, 바닥 있는 균일 depression이 붙는다. 이것이 정확히 LCA이고 고정점이 lasso라서 theorem 0이 그대로 선다.

경쟁은 세 곳에 서로 다른 형태로 있다.
- 연산이 인출인 곳에는 softmax WTA가 있다. frame κ는 위치 키 → 색 값 인출이다(TEM-t, Hopfield/attention).
- 연산이 추론인 곳에는 시냅스 사이 볼록 경쟁(LCA)이 있다. 선택이 유일하고 보장이 있다.
- 연산이 탐색인 곳에는 3요소 조절을 받는 비볼록 WTA가 있다(torus 탐색).

셋의 역할을 가르는 원리는 하나다: 정상 상태에서 무엇이 남는가. 고정점에서는 홀수 로브, eligibility trace, TD 조절 신호가 모두 사라진다. 그래서 rule head에 남는 것은 Hebbian DC + 경쟁 = lasso이고, 보장은 거기에 있다. 시간 비대칭은 경로에 있다: 지각, 수렴 전의 유한 블록, 비볼록 탐색.

TEM의 분해는 이 그림의 공간 쪽이다. φ와 frame은 공유 가중치가 계산하는 구조 부호이고, 색은 내용이다. rule head는 경험(데모)으로만 쓰고, 새 장소(질의)에서 그 장소의 구조 부호로 읽는 기억이다. 그 갱신 (y − r̂)φᵀ는 TEM 기억 갱신 M ← λM + η(p − p̂)(p + p̂)ᵀ와 예측오차 인자 (p − p̂)를 공유한다. pre 인자는 다르다: TEM은 p + p̂, 우리는 φ.

논문에 쓰지 말 것:
- STDP가 rule head의 선택을 개선한다는 주장. 볼록 영역에서는 증명적으로 불가능하다.
- 어떤 경쟁 교체가 ARC-2를 고친다는 주장.
- LCA 읽기가 새롭다는 주장.

## 4. 코드에 넣는 방법

### 4.1 필수 작업 (forward 변경 없음)

- arc/model.py의 모듈 docstring(1–37행)과 statistics()/step() docstring을 LCA·델타 규칙 읽기로 고친다. "soft = 균일 depression + floor", "θG = 측방 억제", "λ=0이면 DeltaNet 꼴"로 적는다.
- tests/test_model.py에 FP64 테스트 두 개를 추가한다.
  - (a) I1: step()의 출력이 soft(S − θ(G − I), λ)와 같은지. frames 경로 포함.
  - (b) P3: φ 고정, lam_learn False, lam_floor 0, θ_0 = 0으로 T블록 돌린 결과가 G의 고유분해로 계산한 S G⁺(I − (I−G)^T)와 1e-10 안에서 같은지.
- 블록 간 compile과 CUDA graph에 영향이 없다.

### 4.2 진단 (CPU, 기존 체크포인트; 개입이 아니라 논문의 배치 영역 서술용)

- 새 스크립트 scripts/analysis/rule_regime.py. 평가 배치에서 블록마다 다음을 잰다.
  - θ의 KKT 잔차를 λ로 나눈 값: support 위 max|e − λ·sign θ|, support 밖 max(|e| − λ)₊
  - φ drift
  - G 스펙트럼: [L,64,64]에 FP64 eigvalsh로 λ_max와 tr/λ_max
  - 질의 칸별 유한-T 인증값 |(I − G_final)^T φ_q|
- 대상 체크포인트: checkpoints/small/b8g4_rope1d_step50000_ema.pt와 runs/small/e1_b8g4_ronly_lam0.0/last.pt.
- 미리 적는 예측: r-only 체크포인트에서 인증값이 큰 칸이 더 자주 틀린다(AUC ≥ 0.65). direct head에서는 classifier(h)가 대부분 답하므로 예측하지 않는다.

### 4.3 선택 arm: rule_kernel (A1을 돌릴 때만)

- 내용: anti-Hebbian 항에 post가 앞서는 로브 하나를 단다. 이 항은 post(예측)가 블록마다 움직이는 쪽이라 정보를 실을 수 있다. 교사 쪽 로브는 y가 상수라서 φ의 인과 필터로 무너지므로 달지 않는다.

```
A_t = (1/n_l) Σ_l' Σ_o w [ r̂_t + c·D_t ] φ_tᵀ / k_a ,   D_t = Σ_{τ≥1} β^τ r̂_{t−τ}  (재귀: D ← β(D + r̂_prev))
k_a = 1 + cβ/(1−β) > 0 을 구조적으로 강제: k_a = softplus(a_raw) (init 3.83 ⇔ c = 0.5), β = 0.95·sigmoid(b_raw) (init 0.85),
c = (k_a − 1)(1−β)/β.  DC가 1로 정규화되므로 P1에 의해 고정점은 정확히 lasso(λ).  θ_{t+1} = soft(θ_t + S_t − A_t − ΣΓC, λ)
```

- 모양: D는 [L, L, O, 13]이고 기존 `others` [L,L]로 마스크한다.
  - ≤10, DDP 4장(rank당 L=16)이면 16·16·121·13·4 B ≈ 1.6 MB다.
  - 단일 GPU L=64면 26 MB다.
  - r̂ = einsum('lcp,mop->lmoc', θ, φ) + einsum('lf,mfoc->lmoc', Γ, κ).
- 명제 4와 조건 N: 다른 레인의 φ, κ는 쓰기 안에서만 읽고, read r은 자기 φ, κ만 쓴다. 따라서 명제 4가 유지된다. 시험 레인은 w = 0이라 조건 N도 그대로다.
- 구현 제약:
  - 새 파라미터는 __init__에서 마지막에 만든다(out_ffn과 같은 방식). 그래야 기존 난수 추출이 바뀌지 않는다.
  - D를 initial_carry, layer 시그니처, segment에 추가한다. first_grad에서 θ, r, gam과 함께 detach하고, 반환할 때도 detach한다.
  - shape는 정적이고, .item()과 호스트 동기화가 없다. FP32 autocast-off 영역 안에 둔다. 그러면 기존 layer_fn compile과 micro-batch CUDA graph에 그대로 들어간다.
  - rule_kernel=False이면 기존 θG 경로를 그대로 둔다(비트 동일).
  - wdmat 설정은 decay_all=False라서 스칼라 파라미터가 감쇠되지 않는다. decay_all=True 설정에서는 train.py의 decay 람다가 NO_DECAY보다 우선하므로 따로 예외를 둬야 한다.
- 테스트:
  - c=0이면 기존 경로와 FP64 허용오차 안에서 같다. 합산 순서가 달라 비트 동일은 아니다.
  - φ를 고정하고 많이 돌리면 FISTA lasso 대비 KKT < 1e-10.
- full ARC로 확장하려면 banded gather가 필요하고 대략 0.5 GB/pass 이상이 든다. 지금은 필요 없다.

## 5. 결정적 실험

모두 단일 실행이고 이진 개입이다. 예측은 실행 전에 scripts/theory/ 아래 .pred 파일로 먼저 적는다. 우선순위는 T1 → T2 → T3 → A1이다.

| ID | 무엇 (단일 이진 스위치) | 미리 적는 예측 | 판정 | 비용 |
|---|---|---|---|---|
| T1 | P1 검증. torus 위치 연산자 특징(wta_torus.py와 같은 512개 인스턴스, 640차원). 선택 방식 2가지 {soft λ=1e-3, hard top-1 IHT (참 sparsity를 아는 유리한 조건)} × 커널 2가지 {끔, 켬(4.3의 로브, c=0.5, β=0.85)}. 모두 θ=0에서 시작해 수렴(KKT < 1e-10, 또는 3,000걸음)까지. 학습 없음. | soft: 커널을 켜도 질의 칸 답이 99.9% 이상 같고, exact도 같다(0.914±0.004). \|θ_on − θ_off\|/\|θ_off\| ≤ 1e-6. hard: 두 커널 조건 모두 exact ≤ 0.30이고, 커널이 선택된 support를 인스턴스의 5% 이상에서 바꾼다. | soft에서 차이가 나면(수렴 확인 후) P1을 논문에 쓸 수 없다. hard에서 차이가 없으면 "시간은 비볼록 선택에서만 답을 바꾼다"를 빼고 "볼록에서는 못 바꾼다"만 남긴다. | CPU 약 10분 |
| T2 | 학습된 코드에서도 ℓ1이 필요한가. 옛 toy zxcyxv/something 레포의 toy_g/ltd_rule.py를 toy/로 옮긴다(학습되는 unitary 기저 Q, 3,072개 인스턴스, 세 그룹). 처음부터 학습. W 단계 {FISTA soft threshold (대조) | floor 없는 sparsemax 쓰기, λ=0, leak 1e-3 ("STDP+WTA로 대체" 조건)}. 반복 수와 학습 걸음은 같다. | 대조: 세 그룹 모두 ≥ 0.93 (문서 값 95.9/94.8/94.5). floor 없음: 처음 보는 변환 그룹 ≤ 0.20, 학습 그룹 ≤ 0.30. | floor 없는 쪽이 대조의 0.05 안이면 "ℓ1은 대체 불가" 주장을 고정 코드로 한정한다. 대조보다 0.5 넘게 낮으면 학습을 넣어도 주장이 선다. | CPU 약 1시간 × 2 |
| T3 | WTA + 3요소가 실제로 일하는 곳(torus 탐색). δ = Δr − b (고정 기준선) vs δ = Δr − v, v ← v + α_v(Δr − v) (인스턴스별 critic, 초기값 b, α_v는 같은 격자로 맞춤). 해에 머물면 Δr = 0이고 v → 0이므로 δ → 0 (TD-STDP의 성질; Frémaux et al. 2010). | 성공 뒤 머무름 0.045 → ≥ 0.80. 성공@64 ≥ 0.93 (현재 0.963). 걸음 중앙값 ≤ 24 (현재 20). | 머무름 < 0.5이면 고정 기준선이 통합 실패의 유일한 원인이 아니다. critic이 감쇠하는 동안 해가 눌려 다시 떠날 위험이 있다. | CPU 수 분 |
| A1 (선택, 사용자 승인 필요) | ≤10, 30/10 1D 고정 RoPE + out FFN + wd(행렬 + λ), frames 8, 80k 걸음, 4×H100 DDP. rule_kernel {끔 | 켬}. 두 arm 모두 새로 돌려 짝을 맞춘다. 블록마다 KKT/λ, \|θ_30 − θ_29\|/\|θ_30\|, 학습된 c와 β를 기록. | 70–80k 창 평균(11회 평가)의 차 \|Δ\| < 5. 참고로 이전 대조는 113.0. 학습 후 c가 \|c\| < 0.05로 붕괴하면 무정보로 선언하고 음성 결과로 읽지 않는다. | \|Δ\| < 5: "규칙 시냅스 커널은 끝단에서 중립, P1과 일치". Δ ≥ +5: 유한-T 경로 효과로, 선택 개선이 아니라 수렴 가속으로 보고. Δ ≤ −5: 끈 채로 유지하고 보고. | 26.8 ms/step × 80k ≈ 36분 + 평가 ≈ 45분/arm. 두 arm 약 1.5시간 ≈ $11 ($6.94/h, 셋업 별도). 구현과 테스트 반나절. |

A1의 한계를 미리 적는다. null 결과는 "무해"와 "무의미"를 가르지 못하고, 논문의 이론 주장은 T1이 결정한다. GPU 작업은 10-28에 끝났고 일정은 Kaggle 코드 11-02, 논문 11-08이다. 그래서 A1은 "끝단에서 시험했다"는 문장 하나가 필요할 때만 돌린다. ARC-2는 돌리지 않는다. 결정 족이 바뀌지 않으므로 0/172로 예측한다.

근거 스크립트는 모두 scratchpad 아래에 있다.
- 이론 검사: scripts/theory/checks.py, scripts/theory/finite_lam.py, scripts/theory/wta_torus.py, scripts/theory/wta_torus_l1.py
- 배치 영역 측정과 Kbar 반례: scripts/review_cdr/spec.py, scripts/review_cdr/kbar.py
- drift, 인증값, 스펙트럼: scripts/review_wta/drift.py, scripts/review_wta/cert.py, scripts/review_wta/gram_spectrum.py

## 6. 위험과 열린 문제

- **이득은 설명 쪽에 있다.** 심사자가 "이름만 바꿨다"고 볼 수 있다. 방어 논리는 이렇다. 항등식(I1–I3)은 정확하다. P1과 P3는 범위(P2)가 명시된 실제 명제다. LCA 읽기 자체가 새롭다고 주장하지 않는다.
- **theorem 0의 전제 C는 배치에서 성립하지 않는다.** 논문 본문에 배치 영역 수치(KKT 10–20λ, φ drift 0.27–0.92, 유효 rank 약 2)와 P3를 함께 적는다.
- **E1의 λ 무효과를 스펙트럼 필터로 설명하는 것은 가설이다.** 측정된 스펙트럼과는 맞는다. direct 8/4의 7점 차(λ=0이 높음)는 단일 seed 잡음보다 크므로 그대로 보고한다.
- **full ARC에서 λ를 없애면 안 된다.** warm start에서 묵은 성분을 지우는 것은 λ뿐이다(2.4의 기능 5).
- **조건 S, ARC-2.** 문헌상 이유가 있는 유일한 지렛대는 목표가 재귀를 따라 변하는 head다. 예를 들어 ψ_t로 φc_{t+1}을 회귀하는 지연 transition 목표(Kappel et al. 2014)가 있다. 비대칭 커널이 rule head에서 진짜 일을 할 수 있는 곳도 여기뿐이다.
  - 같은 깊이의 clamped-code 행은 transition이 아니다.
  - 시험할 때는 목표를 그룹 사이에서 섞은 대조와 clamped 흐름의 stop-gradient로 교란을 막아야 한다.
  - 비용: 지각 계산이 두 배가 되고, 훈련 중 누설 경로가 13차원에서 13+p차원으로 넓어진다.
  - 11-08 전에는 한계와 다음 단계로만 쓴다.
- **조건 N.** 평가에서는 정확하고, 훈련에서는 1차까지만 성립한다. y_A → S_B → θ_B → r_B → W_fb → h_B → φ_B → θ_A 경로는 아직 재지 않았다.
- **고정점 이론의 공백.** 결합 (지각, θ) 고정점의 존재와 도달 가능성은 증명되지 않았다. P1은 도달한 φ*를 전제로 한 진술이다.
- **bands의 tonic 같은 시각 항.** 사용자 결정대로 공학적 세부로 둔다. 이 항을 커널 이론의 일부로 내세우지 않는다.
- **해석으로만 쓸 것.** Sudoku 커널을 "변화를 기록한다"(zero-DC 계열)로 읽는 것과 TEM 연결은 해석이다. 결과로 쓰지 않는다.
- **표기.** bands의 S [D×D]와 rule head의 S [13×p], 구조코드 G와 Gram G가 충돌하므로 논문에서 개명한다.

## 7. 참고문헌

표기:
- [V] 원문 PDF나 로컬 텍스트로 확인
- [Va] 초록이나 웹 서지로 확인
- [U] 기억에 의존, 원문 확인 필요

**경쟁과 sparse coding**
- [V] Rozell, Johnson, Baraniuk & Olshausen 2008, Neural Computation 20:2526–2563 (LCA, eq. 3.2; HLCA는 비볼록)
- [V] Gregor & LeCun 2010, ICML (LISTA, "mutual inhibition / explaining away")
- [Va] Chen, Liu, Wang & Yin 2018, NeurIPS (LISTA의 weight coupling과 선형 수렴)
- [Va] Balavoine, Romberg & Rozell 2012, IEEE TNNLS 23(9) (LCA 수렴)
- [V] Földiák 1990, Biological Cybernetics 64:165–170 (anti-Hebbian 탈상관; WTA 반대 논거)
- [V] Zylberberg, Murphy & DeWeese 2011, PLoS Comput Biol 7:e1002250 (SAILnet; 문턱 = Lagrange 승수)
- [V] Olshausen & Field 1997, Vision Research 37:3311–3325
- [V] Blumensath & Davies, arXiv:0805.0510 (IHT, RIP 조건). [U] ACHA 27(3) 2009 판의 상수 δ_3s < 1/√32
- [V] Donoho & Elad 2003, PNAS 100:2197–2202 (spark, coherence)
- [U] NSP 필요충분 정리의 원 출처: Donoho & Huo 2001; Gribonval & Nielsen 2003; Cohen, Dahmen & DeVore 2009. [V] 고전 결과로 인용한 문헌은 arXiv:2304.05526
- [V] Makhzani & Frey, arXiv:1312.5663 (k-sparse AE = hard thresholding)
- [V] Martins & Astudillo 2016, arXiv:1602.02068 (sparsemax, Prop. 3)
- [U] Beck & Teboulle 2009 (FISTA); Polyak 1964 (heavy ball); Fuchs 2004, Zhao & Yu 2006, Wainwright 2009 (유한-λ support 복원); Landweber 1951, Engl, Hanke & Neubauer 1996, Yao, Rosasco & Caponnetto 2007 (조기 정지 = 정규화); Donoho & Johnstone 1994 (universal threshold)

**STDP, WTA, 3요소 규칙**
- [V] Xie & Seung, NIPS 12 (1999/2000) (DC = Hebb, 1차 모멘트 = pre × d post/dt)
- [V] Bengio, Mesnard, Fischer, Zhang & Wu, arXiv:1509.05936 (2015)
- [Va] Bengio et al. 2017, Neural Computation 29(3):555–577. 2015년 논문과 다른 논문이다.
- [Va] Scellier & Bengio 2017, Front Comput Neurosci 11:24 (Equilibrium Propagation)
- [V] Song, Miller & Abbott 2000, Nature Neuroscience 3:919–926 (경쟁에는 창 적분이 음수여야 함)
- [V] Masquelier & Thorpe 2007, PLoS Comput Biol 3:e31
- [V] Nessler, Pfeiffer, Buesing & Maass 2013, PLoS Comput Biol 9:e1003037. 이들의 −1 depression은 post spike가 촉발하고 바닥이 없다. soft threshold와 같다고 쓰지 말 것.
- [V] Diehl & Cook 2015, Front Comput Neurosci 9:99
- [V] Mozafari et al. 2018, IEEE TNNLS (ar5iv로 확인, 학술지 정보는 목록에서)
- [Va] Maass 2000, Neural Computation 12:2519–2535
- [V] Kappel, Nessler & Maass 2014, PLoS Comput Biol 10:e1003511 (WTA + lateral STDP로 HMM transition 학습)
- [Va] Legenstein, Naeger & Maass 2005, Neural Computation 17(11)
- [V] Frémaux & Gerstner 2016, Front Neural Circuits 9:85
- [V] Frémaux, Sprekeler & Gerstner 2010, J Neuroscience 30:13326–13337 (과제별 critic이 필요함)
- [V] Miconi, Rawal, Clune & Stanley 2019, ICLR (Backpropamine; 가장 가까운 선행 연구)
- [U] Gerstner et al. 2018, Front Neural Circuits 12:53 (초록만 봄); Kempter, Gerstner & van Hemmen 1999; Hopfield 1982

**Fast weight, attention, 맥락 내 회귀**
- [V] Schlag, Irie & Schmidhuber 2021, ICML
- [V] Yang, Wang, Zhang, Shen & Kim 2024, NeurIPS (DeltaNet)
- [V] Ramsauer et al. 2021, ICLR (modern Hopfield = attention)
- [U] Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025); Wang, Shi & Fox, arXiv:2501.12352 (test-time regression)
- [U] von Oswald et al. 2023, ICML ("Transformers learn in-context by gradient descent"); Akyürek et al. 2023, ICLR. rule head = "맥락 위 델타 규칙"이라는 점에서 가장 가까운 ML 선행 연구이고, 논문에 반드시 넣어야 한다.

**TEM**
- [V] Whittington et al. 2020, Cell 183:1249–1263 (STAR Methods의 M ← λM + η(p − p̂)(p + p̂)ᵀ)
- [U] Whittington, Warren & Behrens 2022, ICLR (TEM-t)
