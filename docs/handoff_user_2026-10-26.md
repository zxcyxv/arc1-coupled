# 사용자 체크리스트 (2026-10-26): ARC-AGI-1/2 본 학습

Claude에게 줄 문서는 [handoff_agent_2026-10-26.md](handoff_agent_2026-10-26.md)다. 이 문서는 사용자가 직접 해야 하는 일만 적는다.

## 1. 아침: 인스턴스 준비 (약 30분)

- [ ] Vast에서 **4×H100 SXM, on-demand**로 빌린다.
  - 매물에 "SXM"이 있는지 본다. PCIe나 NVL은 대역폭이 낮아 거의 이득이 없다.
  - 디스크는 50~60 GB로 잡는다.
  - Max CUDA는 12.4 이상이면 된다(setup이 맞는 torch를 깐다).
  - Max Duration은 5일 이상이어야 한다(ARC-1 + ARC-2 약 4일).
  - 이미지는 Vast base 이미지(`/venv/main` 있는 것)로 한다.
- [ ] 인스턴스 터미널에서 `gh auth login`을 한다. 로그와 milestone을 푸시할 때 필요하다.
- [ ] `git clone https://github.com/zxcyxv/arc1-coupled && cd arc1-coupled`
- [ ] Claude를 열고 이렇게 말한다:
  > `docs/handoff_agent_2026-10-26.md`를 읽고 §4 순서대로 진행해. setup, DDP 확장성 bench, 시험 런까지 하고 결과를 알려줘.

## 2. 본 런을 시작하기 전 확인 (Claude 보고를 보고 판단)

- [ ] **setup 통과:** torch가 모든 GPU를 인식하고, torch.compile, NCCL, 테스트가 OK인지.
- [ ] **bench:** 4 GPU s/step이 예측(약 0.65 s, 역전파 4블록)과 크게 다르지 않은지, 예상 일수가 일정에 맞는지. 많이 느리면 다른 매물로 바꿀지 정한다.
- [ ] **DDP 확장성:** 4장 768레인의 s/step이 1장 192레인의 s/step과 같거나 +5% 이내인지.
- [ ] **시험 런:** 평가, 중단, 재개, 최종 평가, 제출 파일이 다 됐는지.
- [ ] 위가 다 되면 ARC-1 본 런 시작을 지시한다(`bash scripts/run.sh start`).

## 3. 학습 중 (하루에 몇 번)

- [ ] 상태 확인: `bash scripts/run.sh status`를 직접 보거나 Claude에게 묻는다(s/step, ETA, GPU, 디스크, 평가).
- [ ] **멈출 시점 결정(사용자 몫):** vote 평가가 5천 에폭(ARC-1 약 3.5시간)마다 나온다.
  - 기준을 미리 정해 두면 좋다. 예: pass@2와 원본 판 exact가 vote 평가 두 번 연속 오르지 않으면 멈춘다.
  - 멈춘 시점과 근거를 적어 둔다. 시험 세트 곡선으로 고른 것이라 논문에 써야 한다.
- [ ] 하루에 한 번 이상 결과를 올린다(Claude에게 "archive하고 푸시해").
- [ ] 문제가 생기면 Claude에게 증상과 `status` 출력을 그대로 준다.
- [ ] 레시피에서 벗어나는 조치는 사용자가 정한다. 예: NaN이 반복될 때 lr을 낮추거나 clip을 켜는 것.

## 4. ARC-1을 마칠 때

- [ ] 멈추기로 했으면 `bash scripts/run.sh stop`. 다음 레인 교체(16스텝 이내)에서 저장하고 종료한다.
- [ ] 최종 평가(1001판, 약 3시간): `CKPT=runs/arc1/step_N.pt bash scripts/run.sh eval`. 끝까지 돌았으면 자동으로 이미 돈다.
- [ ] archive하고 푸시한다(로그, 평가, milestone).
- [ ] **ARC-2 시작:** `RUN=arc2 CONFIG=configs/arc2.json bash scripts/run.sh start`(약 2.7일).

## 5. ARC-2를 마칠 때

- [ ] 같은 절차: 멈추기, 최종 평가(약 1시간), archive, 푸시.
- [ ] GitHub에 결과(`docs/evidence/run_arc1`, `run_arc2`, `checkpoints/run_*`)가 올라갔는지 확인한다.
- [ ] **인스턴스 destroy.** stop만 하면 스토리지 요금이 계속 나간다.

## 6. 따로 할 일

- [ ] 이 5090 인스턴스를 반납할지 정한다. 10-25 밤의 ≤10 절제(역전파 2블록, 블록 8개)는 이미 이 인스턴스에서 돌았다. 기록은 이전 레포 `docs/evidence/`에 있다.
- [ ] 이전 레포(`something`)에 커밋 안 된 변경을 커밋한다: `lt/coupled.py`의 Γ, 테스트, Γ 절제 로그.
- [ ] **논문:**
  - 비교할 TRM 수치를 고른다(ARC-1 44.6% 논문 / 약 40% 재현, ARC-2 7.8%).
  - 추가 분석을 고른다(agent 문서 §8의 후보).
  - TRM과 다른 점 7가지를 본문에 적는다(agent 문서 §8).

## 7. 비용과 기간 (예측)

| 단계 | 기간 (4×H100 SXM, ±30%) |
|---|---|
| setup, bench, 시험 런 | 약 1시간 |
| ARC-1 50k 에폭 | 약 2.0일 |
| ARC-1 최종 평가 | 약 3시간 |
| ARC-2 50k 에폭 | 약 2.7일 |
| ARC-2 최종 평가 | 약 1시간 |
| **합계** | 약 4.9일 |

역전파 4블록 기준이다(10-25에 2에서 되돌림). 시간당 $6.94 매물이면 약 $815(약 108만 원)이고, 일찍 멈추면 그만큼 줄어든다. ARC-1 결과는 시작 후 약 2일 3시간, ARC-2 결과는 약 4.9일 뒤에 나온다.
