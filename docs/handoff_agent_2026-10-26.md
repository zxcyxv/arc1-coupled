# 에이전트 인수인계서 (2026-10-25 → 10-26): ARC-AGI-1/2 본 학습

내일 새 세션의 Claude가 처음 읽는 문서다. 사용자가 직접 할 일은 [handoff_user_2026-10-26.md](handoff_user_2026-10-26.md)에 따로 있다.
읽는 순서: 이 문서 → [decisions.md](decisions.md)(설정과 근거) → [../README.md](../README.md)(명령) → [../AGENTS.md](../AGENTS.md)(규칙).

## 1. 사용자와 일하는 방식

- 한국어로 직접적이고 정확하게 답한다. 틀리면 분명히 말한다. 수리논리적 논증을 선호한다.
- **질문은 지시가 아니다.** "~아닌가?"에는 따져서 답하고, 코드나 문서를 바꾸지 않는다.
- **실행은 기본으로 백그라운드다.** Bash `run_in_background`를 쓰고, 포그라운드 `sleep`은 쓰지 않는다. 학습과 평가는 `scripts/run.sh start|eval`(setsid nohup)로 띄운다. 감시는 실제 python 워커 PID로 하고, `pkill -f`는 쓰지 않는다.
- 이미 한 실험을 확인 명목으로 다시 돌리자고 하지 않는다.
- `docs/decisions.md`의 확정 사항은 사용자 승인 없이 바꾸지 않는다. 바꾸면 문서를 고친다.
- 커밋과 푸시는 사용자가 요청할 때 한다. 10-25에는 요청을 받아 이 레포를 만들고 푸시했다.
- 장기 학습 중 보고는 vote 평가(5천 에폭마다), 오류, 요청받은 시점에만 한다.
- **시간이 가장 큰 제약이다.** 10-25 기준으로 논문 제출까지 1주일이고, 논문은 아직 한 글자도 쓰지 않았다. 무언가를 제안할 때는 걸리는 시간과 비용을 같이 말한다.
- 디스크가 작다(인스턴스 50~60 GB). 큰 임시 파일은 쓰고 바로 지운다.

## 2. 과제

- **모델(coupled):** bands 지각(공유 블록 재귀, 복소 극점 흔적, FP32 W) + 과제별 lasso 규칙 헤드 Θ + Γ(틀 색) 항. 시범에서 질의로 가는 정보는 Θ와 Γ만 거친다.
- **목표:** TRM(SamsungSAILMontreal/TinyRecursiveModels)과 같은 데이터, 증강, 최적화, 배치, loops로 ARC-AGI-1과 ARC-AGI-2를 학습해 비교한다. 같은 것과 다른 것은 decisions.md에 있다.
- **직전 근거(≤10 절제, 278쌍, 15k스텝, 이전 레포 `docs/evidence/`):**

  | 런 | exact 최종 (최고) | 칸 |
  |---|---|---:|
  | Γ, 32블록, 역전파 4 | 69 (72) | 0.868 |
  | Γ, 32블록, 역전파 2 | 54 (54) | 0.857 |
  | Γ, 블록 8개, 역전파 4 | 6k에 50(32블록 역전파 4와 같은 스텝에서 동일). 10-25 밤에 진행 중 | 0.853 (6k) |
  | 기존 coupled(Γ 없음) | 38 (45) | |

  비교 대상인 cmap은 65, URM은 47이다. 이 결과로 역전파를 4로 되돌렸다. 손해는 역전파 수에서 오고 블록 수(8 대 32)에서는 거의 오지 않는다.

## 3. 지금 상태

- **레포:** `github.com/zxcyxv/arc1-coupled`(public). 10-25 작업 전체가 `main`에 있다.

| 확인 항목 | 결과 |
|---|---|
| 모델 = 이전 `lt/coupled.py` | 같은 seed에서 초기 가중치 동일. 출력은 Γ 없으면 비트 단위 동일, Γ 있으면 차이 2.8e-14(FP64) |
| 데이터 = TRM | ARC-1(학습 캔버스 3,633,410 + 시험 385,815)과 ARC-2(5,163,277 + 170,393) 모두 바이트 단위 동일 |
| 옵티마이저 = TRM | `adam-atan2 0.0.3` 커널 식과 원소별 동일 |
| DDP | CPU 2프로세스 = 1프로세스 grad_accum 2(가중치 차 ≤ 1e-12). 재개는 비트 단위로 정확 |
| 운영 흐름 | 5090에서 bench, start, status, stop, 재개, quick/vote/final 평가, 제출 파일, 따로 돌리는 eval까지 통과 |
| 단위 테스트 | 26개 통과 |

- **실측 (5090, 400 W 전력 제한, Γ 포함, BF16 틀):**
  - 레인당: 역전파 4블록 6.33 ms/step, 역전파 2블록 4.79 ms/step
  - 메모리(GPU당 192레인, 역전파 4): grad_accum 2로 약 60 GiB. grad_accum 1은 H100 80 GB를 넘는다
  - Γ 틀 attention을 BF16으로 하면 FP32보다 10.6% 빠르다
- **예측 (4×H100 SXM, 역전파 4, 대역폭으로 환산, ±30%):**
  - 약 0.65 s/step
  - ARC-1 50k 에폭(259,113스텝) 약 2.0일, ARC-2 50k 에폭(361,049스텝) 약 2.7일
  - 1001판 최종 평가: ARC-1 약 3시간, ARC-2 약 1시간
- **확인하지 못한 것(오늘 드러남):**
  1. 4 GPU에서의 DDP 속도 확장. 여기는 GPU가 한 장뿐이었다.
  2. cu126 휠. 5090은 cu128이 필요해서 시험하지 못했다.
  3. H100 실제 속도.
  4. 16 loops와 carry의 효과. ≤10 절제는 모두 loops 1(Θ가 매 스텝 0에서 시작)이었다. 다만 10-25 밤에 5090에서 내일 하네스를 ARC-1 전체로 돌렸다(`runs/arc1_5090_g2`, 배치 64, 역전파 2). 학습은 됐다: 6,224스텝에 평가 칸 0.802, exact 1/419. 학습 배치 칸은 약 0.85에서 평평했다(작은 배치 탓일 수 있음).

## 4. 오늘 순서

사용자가 일부를 직접 할 수 있으니, 어디까지 했는지 먼저 묻는다.

1. `bash scripts/run.sh setup`. 기대 결과:
   - torch 2.11.0+cu126(드라이버가 CUDA 12.8 미만) 또는 cu128
   - "matmul + torch.compile ok on every GPU"
   - NCCL ok(실패하면 setup이 P2P를 끄고 다시 시도해 `runs/env.sh`에 기록)
   - sha OK 2개, 테스트 26개 OK
2. **DDP 확장성 점검:** `NGPU=1 BENCH_GA=2 EXTRA="global_batch_size=192" bash scripts/run.sh bench`를 먼저 돌리고, `bash scripts/run.sh bench`를 마지막에 돌린다(이게 `runs/bench.json`을 쓴다). 4장의 s/step이 1장 192레인의 s/step보다 5% 넘게 크면 원인을 찾는다(§5).
3. **시험 런 (4 GPU, 약 10분):**
   ```bash
   export TRIAL="max_steps=192 quick_eval_epochs=10 vote_eval_epochs=20 vote_versions=2 save_every_steps=64 final_versions=2"
   RUN=trial EXTRA="$TRIAL" bash scripts/run.sh start
   ```
   100스텝 무렵 `RUN=trial bash scripts/run.sh stop`으로 멈추고, 같은 명령으로 다시 시작해 재개되는지 본다. 끝나면 `eval.jsonl`에 quick(64), vote(112), quick(160), final(192)이 있고 `submission_step_192.json`이 있어야 한다. 확인하면 `rm -r runs/trial`.
4. **ARC-1 본 런:** `bash scripts/run.sh start`. 첫 30분 체크는 §5.
5. **ARC-1 마무리:**
   - 사용자가 멈추라고 하면 `stop`, 그다음 `CKPT=runs/arc1/step_N.pt bash scripts/run.sh eval`(1001판, 약 3시간). 끝까지 돌면 최종 평가가 자동으로 붙는다.
   - 그다음 `archive`. 사용자가 요청하면 커밋하고 푸시한다(`git add docs/evidence/run_arc1 && git add -f checkpoints/run_arc1/*.pt`).
6. **ARC-2:** ARC-1 최종 평가가 끝난 뒤 `RUN=arc2 CONFIG=configs/arc2.json bash scripts/run.sh start`. 나머지는 같다.

## 5. 모니터링과 대응

**첫 30분**
- [ ] s/step이 bench 값의 ±10% 이내인가.
- [ ] GPU 이용률이 약 99%이고 메모리 피크가 75 GB 이하인가.
- [ ] `clocks_event_reasons`: `0x4`(전력 상한)는 괜찮다. `0x8`, `0x20`, `0x40`이 계속 뜨면 사용자에게 알린다.
- [ ] 손실이 줄어드는가. lr은 2,000스텝까지 워밍업한다.
- [ ] 2,048스텝에 `step_2048.pt`가 생기는가.

**주기적으로**
- [ ] vote 평가(약 3.5시간마다)의 pass@1/2, `view_exact`, 원본 판 exact를 사용자에게 짧게 보고한다. TRM 곡선(§7)과 같은 에폭에서 비교한다.
- [ ] 디스크(`df -h`)를 본다.
- [ ] 사용자가 원하면 archive하고 푸시한다.

**증상별 대응**

| 증상 | 대응 |
|---|---|
| `FloatingPointError`(비유한 손실이나 기울기)로 정지 | 마지막 체크포인트는 그대로 있다. 같은 명령으로 재개해 본다. 같은 곳에서 반복되면 사용자에게 알리고, lr 낮추기나 clip 같은 레시피 이탈은 사용자가 정한다 |
| OOM | `GA=4 bash scripts/run.sh start` |
| 진행 정지 | NCCL을 의심한다. 워커 PID를 kill하고 `echo 'export NCCL_P2P_DISABLE=1' > runs/env.sh` 후 재개 |
| bench보다 느림 | throttle 이유, 다른 프로세스, `nvidia-smi topo -m`을 본다 |
| DDP 확장성 불량 | 기울기 all-reduce(15 MB/스텝)가 느린지 `NCCL_DEBUG=INFO`로 전송 경로를 확인한다. P2P나 SHM이 없으면 확인한다 |
| cu126에서 컴파일 오류 | `EXTRA="compile=false"`로 돌린다(느림). 사용자에게 알린다 |
| EXITCODE가 0이 아님 | `stdout.log` 끝에서 원인을 잡고 같은 명령으로 재개 |

## 6. 사용자가 정하는 것 (먼저 정하지 말 것)

- 멈출 시점. 근거가 될 곡선을 정리해 주는 것까지가 우리 몫이다.
- 레시피 이탈(lr, clip, 블록 수 등)과 인스턴스 교체.
- 추가 분석을 할지와 무엇을 할지.

## 7. 기준 수치

- **TRM ARC-1 곡선**(`all/exact_accuracy`, 에폭): 10k 0.004, 20k 0.054, 30k 0.141, 50k 0.226, 100k 0.264. 이전 레포 `docs/evidence/trm_arc1/reference.json`에 있다. 우리 쪽에서 가장 가까운 지표는 vote 평가의 `view_exact`(8판 평균)다.
- **보고된 수치:** TRM ARC-1 pass@2 44.6%(논문), 재현 약 40%, ARC-2 약 7.8%. URM은 ARC-1 53.8%, ARC-2 16.0%(pass@1).

## 8. 논문용 산출물

- **런마다:** `eval.jsonl`, 최종 평가 json과 제출 파일, `train.jsonl.gz`, `config.json`, GPU 종류, s/step, 총 시간.
- **TRM과 다른 점(본문에 적어야 함):**
  - 역전파 4/8블록(TRM 10/30)
  - ACT 없음(16 고정, 동시 교체)
  - Γ 헤드
  - 50k 에폭 또는 수동 중단(TRM 100k)
  - 단일 체크포인트 투표(TRM은 10번 누적)
  - 동률을 시범 정합도로 깸(TRM은 q_halt)
  - Γ 틀 읽기 BF16
- **싼 분석 후보(학습 뒤, milestone으로 가능):**
  - 시범 개수 스케일링
  - 추론 세그먼트 곡선(`--segments 8/16/32`)
  - 평가 때 Γ = 0 절제
  - 격자 크기별 분해(≤10 / >10)
  - ≤10의 역전파 수(2 대 4)와 블록 수(8 대 32) 절제는 10-25에 했다(§2). 논문의 절제 표로 쓸 수 있다

## 9. 이전 레포(`zxcyxv/something`) 상태

- **커밋 안 된 변경:** `lt/coupled.py`(Γ = frames), `lt/coupled_train.py`(gamma_abs 기록), `tests/test_coupled.py`(Γ 테스트 3개), `docs/evidence/run_coupled_arc1_d512_every1_frames8_s0/`.
- 이 레포의 모델은 그 `lt/coupled.py`와 같은 출력을 낸다(§3).
