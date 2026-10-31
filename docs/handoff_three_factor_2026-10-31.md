# 인수인계: 3요소(TD) 비교 실험 (2026-10-31)

GPU 인스턴스에서 새 세션을 열어 이 실험을 이어 가는 Claude와 사용자가 읽는다. 코드는 다 짜여 있고 CPU 테스트를 통과했다.
GPU에서는 한 번도 돌리지 않았다. 학습 결과는 아직 없다.

## 0. 한눈에

- **묻는 것:** 결과에 따라 부호가 바뀌는 W 항(세 번째 요인 = 예측된 TD 오차)을 bands EMA 모델에 더하면, 학습 깊이를 넘긴
  외삽에서 오차 감소율이 오르는가.
- **과제:** 스도쿠-Extreme 1k(`data/sudoku_lt_1k.npz`, 학습 1,000 퍼즐 × 증강, 평가 2,048 퍼즐).
- **모델:** 이전 레포의 스도쿠 bands EMA 모델을 폭 512로 줄인 것. 16블록 전부 역전파 × 16세그먼트 = 퍼즐당 256블록.
- **판:** `bands`(기준), `tf_zero`(3요소, W가 퍼즐마다 0에서 시작), `tf_w0`(3요소, W가 학습된 W0에서 시작). 대조판
  `w0`(3요소 없이 W0만)는 필요할 때만 돌린다. 판끼리는 플래그만 다르다.
- **주 지표:** 5천 스텝마다 128세그먼트(학습 깊이의 8배) 외삽의 오차 감소율.
- **코드 위치:** 브랜치 `claude/clever-rubin-bw64wm`. main에는 아직 없다.

## 1. 새 세션 시작

```bash
cd /workspace && git clone https://github.com/zxcyxv/arc1-coupled && cd arc1-coupled
git checkout claude/clever-rubin-bw64wm
mkdir -p ~/.claude/projects/-workspace-arc1-coupled/memory
cp docs/agent_memory/*.md ~/.claude/projects/-workspace-arc1-coupled/memory/
rm -f ~/.claude/projects/-workspace-arc1-coupled/memory/README.md
bash scripts/tf_runs.sh setup
```

- 레포 경로가 다르면 메모리 폴더 이름을 그 경로(슬래시 → `-`)로 바꾼다.
- `setup`이 하는 일:
  - GPU 목록을 보여 준다.
  - CUDA torch가 없으면 설치한다(드라이버에 따라 cu128 또는 cu126).
  - 데이터 체크섬을 확인한다.
  - `tests.test_lt_three_factor`를 돌리고 디스크 상태를 보여 준다.
- 먼저 `AGENTS.md`와 메모리의 사용자 규칙(seed 하나, 이분법 실험, 워크플로 금지)을 읽는다.

## 2. 실행 순서

### 2.1 짧은 벤치 (본 런 전에, 백그라운드)

가장 무거운 판(`tf_w0`)과 기준(`bands`)을 각각 96스텝씩 돌려 s/step과 메모리를 잰다. 컴파일 때문에 처음 몇 스텝은 느리다.

```bash
mkdir -p runs/tf_smoke && CUDA_VISIBLE_DEVICES=0 setsid nohup python -m lt.train --config configs/tf/base.json --out runs/tf_smoke/tf_w0 --set three_factor=true w0_learned=true max_steps=96 milestone_every=0 save_final=false log_every=16 > runs/tf_smoke/tf_w0.log 2>&1 < /dev/null &
```
```bash
mkdir -p runs/tf_smoke && CUDA_VISIBLE_DEVICES=1 setsid nohup python -m lt.train --config configs/tf/base.json --out runs/tf_smoke/bands --set max_steps=96 milestone_every=0 save_final=false log_every=16 > runs/tf_smoke/bands.log 2>&1 < /dev/null &
```

- `runs/tf_smoke/*/train.jsonl`의 `seconds`(스텝 시간)와 `mem_gb`(최대 메모리)를 읽는다. 끝나면 `runs/tf_smoke`를 지운다.
- **메모리가 넘칠 때:** 모든 판에 똑같이 `EXTRA="grad_accum=2"`를 준다. `grad_accum`은 재개 조건(RECIPE_KEYS)이라,
  런 도중에는 바꿀 수 없다.
- **본 런 시간 추정:** 30,000 × s/step + 평가 15회(2,048 퍼즐 × 16세그먼트) + 외삽 6회(512 퍼즐 × 128세그먼트).
  사용자에게 보고하고 지시를 받은 뒤 본 런을 띄운다.

### 2.2 본 런

```bash
bash scripts/tf_runs.sh start bands tf_zero tf_w0
```
```bash
bash scripts/tf_runs.sh status
```

- 목록의 k번째 판이 GPU k에 뜬다. 다른 GPU를 쓰려면 `GPUS="1 2 3"`처럼 지정한다.
- 로그는 `runs/tf/<판>/stdout.log`, `train.jsonl`, `eval.jsonl`, `milestones/`에 쌓인다.
- 감시는 `status`가 보여 주는 python 워커 PID로 한다. `pkill -f`는 쓰지 않는다.
- **멈추기:** `bash scripts/tf_runs.sh stop <판>`. SIGTERM을 받으면 체크포인트를 쓰고 끝난다.
- **재개:** 같은 명령으로 다시 `start`한다. 같은 `--out`이라 이어서 돈다.
  - 설정(RECIPE_KEYS)이 다르면 재개를 거부한다. `max_steps`는 바꿀 수 있다.
  - 체크포인트에는 carry가 없다(`save_carry: false`, AGENTS.md 규칙). 2,000스텝마다 저장하고, 2,000은 16의 배수라 모든
    레인이 교체되는 시점과 맞는다.
  - 퍼즐 중간에 멈췄다 재개하면 레인이 새 퍼즐로 시작한다.
- **실행 중인 런의 코드와 설정은 바꾸지 않는다.**
- **디스크:** 판마다 체크포인트 약 70 MB × (최근 2개 + 마일스톤 6개 + 마지막), 로그 수십 MB.
- **Vast 주의:** 인스턴스를 destroy하면 전부 사라진다. 남길 로그는 5절대로 레포에 옮겨 커밋한다(커밋은 사용자 요청 시).

### 2.3 보고

- vote 평가에 해당하는 것은 외삽 마일스톤(`[EXTRAP]`, 5천 스텝마다)이다.
- **보고하는 때:** 마일스톤마다, 오류가 났을 때, 사용자가 물을 때. `[EVAL]`(약 1,953스텝마다)마다 보고하지는 않는다.
- 초반 평가의 exact가 낮은 것은 이 모델 계열에서 정상이다(메모리 early-generalization-gap). 그것만 보고 설정을 바꾸자고 하지 않는다.
- **보고 형식:** 판별로 같은 스텝의 수치를 나란히 놓는다.
  - 외삽: seg16 exact/512, 최고(세그먼트 번호), 오차 감소율.
  - `[EVAL]`: exact/2048.
  - 3요소 판만: β, λ, δ와 Ĉ − CE(seg16과 seg128).

## 3. 모델 (n = 퍼즐 시작부터 센 블록 번호, 세그먼트를 넘어 이어짐)

기존 bands 쓰기 S_n은 그대로다. 그 위에 다음이 더해진다(헤드 h마다).

```
C_n  = λ_h C_{n-1} + (1 - λ_h) S_n                                    적격성 흔적 [B,H,D,D]
W~_n = (1 - η_h) W~_{n-1} + η_h S_n + η_h β_h δ_n C_{n-1}             β = 0이면 지금 모델
W_n  = W~_n + (1 - η_h)^n W0        (w0_learned; 아니면 W_n = W~_n)
read = q~ (Σ_t v k~^T + W_n)^T                                        (hebb, 지금과 같음)
δ_n  = -(1 - γ) Ĉ(h_{n-2}) + γ V(h_{n-1}) - V(h_{n-2}),   δ_1 = 0      (h_{n-1} = 블록 n의 입력)
```

- **신용은 지연 1로 들어간다.** δ_n은 블록 n−1이 만든 전이를 평가하므로, 블록 n−1의 쓰기를 담은 C_{n−1}에 곱한다.
- **헤드:** 칸마다 [Ĉ_t, V_t] = Linear(d→128) → SiLU → Linear(128→2)이고, Ĉ_t = softplus. 입력은 분리된 h_t
  (prefix 토큰 제외)이고, 보드 값은 칸 평균이다.
- **헤드 손실 (세그먼트마다):** 상태 s_0..s_nb는 블록 입력과 출력이다. j < nb마다
  - Ĉ_t(s_j) → CE_t(s_j)(라벨로 계산한 칸별 stablemax CE, MSE)
  - V_t(s_j) → −(1−γ)Ĉ_t(s_j) + γV_t(s_{j+1})(두 목표 모두 detach, MSE)
  - j 평균, 칸 평균, 레인 합. 주 손실에 더하지만 입력이 분리돼 있어 헤드에만 닿는다.
  - 세그먼트 경계를 넘는 전이는 다음 세그먼트에서 carry.hidden(= s_0)으로 학습된다.
- **δ:** 언제나 detach. 학습과 평가에서 같은 예측 δ를 쓴다(평가에도 라벨이나 외부 판정을 쓰지 않는다).
- **초기값:**
  - β = 0(부호 있음, 헤드마다)이라 시작 시점의 모델은 기존 모델과 같다.
  - λ = 0.75(τ = 4블록, 헤드마다 학습).
  - 헤드 출력 가중치는 0이고, bias (0, −ln 2)라 δ가 0에서 시작한다.
  - W0 = 0.
- **학습:** β와 λ는 주 손실로 배운다(1차원이라 weight decay 없음).
  - 헤드 행렬에는 다른 행렬과 같이 wd 1.0이 걸린다.
  - W0에는 wd가 없다(`w0_decay: false`). `init_hidden`처럼 초기 상태로 본다.
- **W0:** 퍼즐 식별자마다 H·D² 크기의 임베딩이다. 스도쿠는 식별자가 하나라 행렬 하나를 공유한다.
  - 위의 분해로 계산하기 때문에 no-grad 구간과 무관하게 모든 grad 블록에서 기울기를 받는다.
  - 몫 (1−η)^n은 η = 0.125면 16블록 뒤 0.12, 32블록 뒤 0.014라서, 사실상 첫 1~2세그먼트에만 작용한다. η가 학습으로
    작아지면 더 오래 남는다(`[ETA]` 줄).
- **탐색 잡음** `tf_noise`(v에 σ·ε·rms(v), 학습 중만)는 기본 0이다. 켜면 기준판을 포함한 모든 판에 같이 켠다.

**누가 정했나**
- **사용자가 정한 것:**
  - 고전 3요소 형식(Izhikevich 2007)을 그대로 취한다.
  - 추론기의 보상은 외부 오라클이 아니다(규칙 검사 기각).
  - 학습된 CE 예측 헤드 Ĉ와 TD로 배우는 V를 쓴다.
  - 학습과 추론 모두 예측 δ를 쓰고, δ는 detach한다.
  - 두 판: W를 0에서 시작, 학습 임베딩을 reshape한 W0에서 시작.
- **Claude가 정한 것 (바꿀 수 있음):**
  - V의 TD 보상에 실제 CE가 아니라 Ĉ를 쓴다. Ĉ가 덜 맞아도 학습 분포에서 δ 평균이 0이 되게 하려는 것이다.
  - 헤드를 칸별로 둔다(폭 128).
  - γ = 1 − 1/32.
  - τ는 4에서 시작해 학습한다.
  - V bias를 −ln 2로 둔다.
  - W0에 wd를 걸지 않는다.
  - 잡음은 기본으로 끈다.
  - 폭 512, 3만 스텝.

## 4. 코드 지도

`lt/`는 이전 레포 `zxcyxv/something@093911e`의 `lt/`(Kaggle 스도쿠 셀이 품은 코드와 같다)에 플래그를 더한 것이다. 이전
세션에서 플래그를 끈 경우(EMA, grad_blocks, hebb off, product) 원본과 logits·기울기가 비트 단위로 같음을 확인했다. 그 비교
스크립트는 원본 레포가 있어야 해서 이 레포에는 없다.

- **`lt/model.py`:**
  - `ModelConfig`: `three_factor`, `tf_gamma`, `tf_elig_tau`, `tf_elig_learn`, `tf_head_dim`, `tf_head_weight`,
    `tf_noise`, `tf_noise_eval`, `w0_learned`, `ce`, `ignore_label0`.
  - `Carry.tf_elig`, `Carry.tf_prev`.
  - `Block.tf_beta`, `tf_lam_raw`, `tf_lam()`.
  - `LT.tf_heads`, `w0_table`.
  - `LT.tf_begin`(세그먼트 시작 상태), `tf_input`(블록마다 δ, 블록 번호, 잡음), `tf_finish`(헤드 손실과 통계),
    `tf_head`, `tf_cell_ce`.
  - `bands_step(..., tf)`와 `tf_write`(위 식). `block_step`은 state의 7번째 칸으로 tf를 넘긴다.
- **`lt/train.py`:**
  - `DEFAULTS`와 `RECIPE_KEYS`(+ `w0_decay`).
  - `METRICS`: `tf_c_loss`, `tf_v_loss`, `tf_delta`, `tf_delta_abs` 추가.
  - `TrainStep`이 `model.tf_loss`를 더한다.
  - `create_optimizers`: W0 wd 제외.
  - `run_segments`(행 3~6: δ, |δ|, Ĉ, 실제 CE), `evaluate`(`[TF]` 줄), `extrapolate`(JSON과 `[EXTRAP-TF]` 줄).
  - `validate`: three_factor를 compile과 쓰면 `compile_scope block`이 필요하다.
- **`tests/test_lt_three_factor.py` (10개, CPU FP64):**
  - β = 0, W0 = 0이면 기존 모델과 같고, δ도 0에서 시작한다.
  - W0 판은 "W = W0에서 시작한 EMA"와 값·기울기가 같다.
  - 3블록 × 2세그먼트 = 6블록 × 1세그먼트(carry가 맞다).
  - δ가 위 공식대로 나오고, 쓰기 식도 맞다.
  - 헤드 손실은 헤드에만 가고, CE는 β와 λ에 간다.
  - 하네스 재개가 정확하다.
- **`configs/tf/base.json`:** 공통 설정. `scripts/tf_runs.sh`: setup / start / status / stop.
- **지원 안 함:** band_product(product 형식), frame_rope, ctx_packed. 켜면 오류가 난다.
- CPU에서 컴파일된 블록(`compile_scope block`)과 eager가 FP32 오차 1e-6으로 같음을 확인했다. GPU 컴파일은 확인하지 않았다.

## 5. 끝나면 할 일

- 판별 로그(`stdout.log`, `train.jsonl`, `eval.jsonl`, `config.json`, `milestones/*.json`, 체크포인트 제외)를
  `docs/evidence/tf_<판>/`으로 복사한다.
- `docs/decisions.md`의 "3요소 비교 실험" 행에 결과와 상태를 적는다.
- 사용자가 요청하면 커밋하고 푸시한다.

## 6. 로그 읽는 법

- **`train.jsonl` (스텝마다):**
  - `lm_loss`
  - `accuracy`, `exact_accuracy`: 16세그먼트를 마친 레인 기준.
  - `tf_c_loss`, `tf_v_loss`: 레인 평균.
  - `tf_delta`, `tf_delta_abs`: 세그먼트 블록 평균의 레인 평균.
  - `seconds`, `mem_gb`.
- **`[EVAL]`:** EMA 가중치, 새 carry, 16세그먼트로 2,048 퍼즐의 exact.
- **`[TF]` (3요소 판):** 헤드별 β와 λ(EMA 가중치), 16세그먼트 출력에서의 δ, |δ|, Ĉ와 실제 CE(2,048 평균).
- **`[EXTRAP]`:** `seg16 X/512 -> best Y (seg k), error reduction (Y − X)/(512 − X)`. Y는 라벨로 고른 세그먼트 하나의
  exact이고, 모든 퍼즐에 같은 세그먼트가 적용된다.
- **`[EXTRAP-TF]`:** seg16과 seg128의 δ, |δ|, Ĉ − CE.
- **`milestones/extrap_step_N.json`:** 세그먼트별(128개) `exact`, `accuracy`, `churn`(직전 세그먼트와 예측이 바뀐 칸
  비율), `delta`, `delta_abs`, `chat`, `ce`.
- **`[BAND]`, `[ETA]`:** 대역의 창 요약과 η(기존과 같음).

## 7. 결과를 읽는 방법 (제안, 판정 규칙은 사용자와 정한다)

seed는 하나만 돌린다(사용자 규칙). 차이는 같은 런의 마일스톤 사이 흔들림과 견주어 본다. 이 작은 모델에는 과거 기록이 없으므로
기준판(`bands`)의 마일스톤 간 변동이 그 기준이다.

**bands 대 tf_zero (주 질문)**

| 관찰 | 해석 |
|---|---|
| β가 0 근처에 머물고 두 판의 곡선이 겹친다 | 학습 구간 안에서 모델이 이 항을 쓸 이유를 찾지 못했다. 외삽 기제에 대해서는 말할 수 없다 |
| β가 움직이고, 같은 마일스톤에서 tf_zero의 오차 감소율이 변동폭보다 높다 | 지지. 기제 확인: seg16 뒤 막힌 퍼즐 구간에서 δ가 음수인지(아래 기제 2) |
| β가 움직이고, 오차 감소율이 낮거나 푼 퍼즐을 잃는다(최고 뒤 exact 하락, churn 증가) | 해롭다. seg16 뒤 Ĉ − CE가 벌어지는지(헤드가 깊은 상태에서 어긋남) 본다 |

**tf_zero 대 tf_w0**
- 질문: 퍼즐 시작의 W를 학습된 사전으로 두는 것이 차이를 만드는가.
- 예측: W0의 몫이 첫 1~2세그먼트에서 사라지므로, 외삽까지 차이가 간다면 초반 궤적을 통해서다.
- 두 판이 다르면 `w0` 대조판(3요소 없이 W0만)으로 W0 단독 효과와 상호작용을 가른다. GPU가 하나 더 필요하다.

**기대하는 기제 (왜 기존보다 나을 수 있나)**
1. 기존 W는 결과를 모른 채 쓰인 것을 같은 비율로 쌓는다(토닉 도파민이 상수인 경우). 그래서 고정점 근처에서는 현재 답을
   굳힌다. 3요소 항은 같은 쓰기라도 결과에 따라 부호가 바뀐다.
2. 막혔을 때(s' = s) δ = −(1−γ)(Ĉ + V)이다. 학습 분포에서 이런 상태가 보통 나아졌다면 V > −Ĉ이고, δ < 0이다. 그러면
   현재 상태를 붙잡는 최근 쓰기가 빠진다. 기대대로 나아가면 δ ≈ 0이고, 풀린 퍼즐은 Ĉ ≈ V ≈ 0이라 건드리지 않는다.
3. δ는 블록 번호를 모르는 헤드가 상태만 보고 낸다. 그래서 256블록이 지나도 같은 규칙이 계속 작동한다. 전제는 헤드가 깊은
   상태에서도 맞는 것이다.
4. Ĉ는 라벨 없는 자기평가라서, 퍼즐마다 세그먼트를 고르는 데 쓸 수 있다. 아직 구현하지 않았다.

## 8. 배경: 고전 시뮬레이션 결과 (이전 세션, FP64, 우리 bands 식)

자세한 내용은 보고 페이지에 있다: https://claude.ai/artifact/9ZkJG1omxktUdmUmGWuik1 (A의 최종 결과는 아직 반영되지 않았다).
시뮬레이션 스크립트는 이전 클라우드 세션의 임시 폴더에만 있고 레포에는 없다.

- **형식 대응:** Izhikevich의 미분방정식을 RK4로 푼 결과와 우리 재귀의 상대 오차가 1.3e-14이다. 지금 EMA 모델은
  λ = 0, 도파민이 상수(= η)인 경우이고, 3요소가 없다.
- **A. 지연 보상** (Izhikevich Fig. 1: 시냅스 10⁴개 중 하나, 보상 1~3초 늦게, 블록 = 10 ms):
  - 1시간(보상 약 34회): 3요소 순위 중앙값 33, 1위 14%, z 2.59. 예측 오차판은 순위 11, z 2.94. 다른 seed도 같았다.
  - 4시간: 1위 96%(예측 오차판 100%).
  - 대조군은 전부 우연 수준이었다: 무작위 시점 보상, λ = 0, λ = |z|², 지금 EMA, product.
  - 학습된 스도쿠 창도 된다. 시차 1에서 음수인 창(head 7)은 보상받은 시냅스를 약하게 만든다(25번 중 22번, z −6.70).
    방향은 도파민 × 창의 부호가 정하므로, 헤드마다 β 부호를 학습해야 한다.
  - 적격성 시정수는 지연의 약 2배가 최적이다(1시간 1위 비율: 0.25 s 실패, 1 s 16%, 2 s 76%, 4 s 96%, 8 s 92%).
  - **조밀도 문제:** 부호 있는 조밀한 활동(p = 1)에서는 같은 보상 수로 z가 0.17에 그쳤다. 무관한 흔적이 모든 시냅스에
    차기 때문이다. 칸 4개가 같은 W에 써도 z가 2.59에서 1.71로 준다. 우리 모델은 81칸 모두 조밀하다.
- **B. 예측 오차와 재귀:**
  - 날 보상을 쓰면 표류만 하고 목표를 고르지 못한다(Frémaux 2010 식 17의 평균 × 평균 항). 평균을 빼면 고른다.
  - 재귀 안에서는 전역 평균도, 블록별 평균도 실패했다. 상태별 critic의 TD 오차만 진전을 낳는 전이를 배웠다
    (테스트 cos 0.17 → 0.94, 블록별 E[δ] ≤ 0.001).
  - 잡음 σ = 0인 결정론 재귀는 아무것도 배우지 못했다(W = 0).
- **우리 실험과 다른 점:**
  - 시뮬레이션에서는 W가 수천 에피소드에 걸쳐 3요소 규칙 자체로 학습됐다.
  - 우리 모델에서는 W가 퍼즐마다 리셋되고, 퍼즐 사이의 학습은 역전파(β, λ, 헤드)가 맡는다.
  - 그래서 σ = 0 실패가 그대로 옮겨지지는 않는다. 퍼즐마다 궤적이 달라 V가 퍼즐별로 틀리는 만큼 δ가 생긴다.
  - 다만 V가 너무 잘 맞으면 δ → 0이 되어 항이 꺼진다. 그때는 `tf_noise`가 수단이다.

## 9. 위험과 열린 문제

- **흔적이 조밀하다(8절 A).** 연속값 뉴런에 맞는 고전 형식은 node perturbation(Williams 1992, Fiete & Seung 2006)이다.
  적격성을 활동 전체가 아니라 요동 ξ로 만든다(C ← λC + (1−λ)Σ_t ξ k̃ᵀ). 구현하지 않았다. 이번 비교 뒤 후보다.
- **헤드가 깊은 상태에서 어긋날 수 있다.** V가 낙관적으로 어긋나면 음의 δ가 계속 나와 W를 지울 수 있다. 마일스톤 JSON의
  δ와 Ĉ − CE로 본다.
- **V가 막힌 상태를 "원래 안 풀리는 상태"로 배우면,** 막혔을 때도 δ ≈ 0이라 기제 2가 작동하지 않는다.
- **발산 위험은 작다(추론).** h는 블록마다 RMSNorm되고 S와 C는 유계라 W도 유계다. 시뮬레이션에서 v 정규화 없이 W가
  발산했던 것과 같은 되먹임은 없다. `train.jsonl`의 `lm_loss`와, 손실이나 기울기가 유한하지 않을 때 하네스가 내는 `FloatingPointError`로 확인한다.
- **속도와 메모리는 재지 않았다.** 헤드 손실 때문에 세그먼트의 블록 입력(16 × B × 81 × 512)을 모아 둔다.
- **Kaggle 런과 직접 비교하지 않는다.** Kaggle의 EMA 42/16 런과 product 런은 폭 832·세그먼트당 42블록이다. 이 실험은
  폭 512·16블록이라, 오차 감소율 수치를 그 런들과 나란히 놓지 않는다.
