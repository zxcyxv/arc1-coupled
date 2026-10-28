# 여기서 시작 (2026-10-28 기준)

새 인스턴스에서 레포를 받은 뒤 가장 먼저 읽는 문서다. 사람과 Claude 둘 다 읽는다.

## 1. 새 세션 첫 명령 (사용자가 Claude에게 그대로 붙여 넣는다)

```
START_HERE.md를 읽고 2절의 메모리 복원과 3절의 문서를 순서대로 읽어. 다 읽으면 현재 상태, 마감,
남은 결정을 10줄 이내로 요약하고, 내가 정할 것만 물어. 학습이나 GPU 작업은 내가 말하기 전에 띄우지 마.
```

그다음 지시는 짧게 해도 된다. 예: "6절 1번 진행해", "10/10 다시 띄워", "30/10 전체 런 capped_run으로 띄워".

## 2. 환경과 메모리 복원 (Claude가 한다)

```
cd /workspace && git clone https://github.com/zxcyxv/arc1-coupled && cd arc1-coupled
mkdir -p ~/.claude/projects/-workspace-arc1-coupled/memory
cp docs/agent_memory/*.md ~/.claude/projects/-workspace-arc1-coupled/memory/
rm ~/.claude/projects/-workspace-arc1-coupled/memory/README.md
bash scripts/run.sh setup        # torch 확인, GPU, NCCL, 데이터 체크섬, 단위 테스트
```

- 레포 경로가 다르면 메모리 폴더 이름의 `-workspace-arc1-coupled`를 그 경로(슬래시 → `-`)로 바꾼다.
- 데이터(`data/*.npz`)와 주요 EMA 가중치(`checkpoints/`)는 레포에 있다. 런별 전체 체크포인트(옵티마이저 포함)와 TRM 공개
  체크포인트(`runs/trm_ckpt`, arcprize 공개본)는 없다.
- Vast 인스턴스는 볼륨이 없으면 destroy 때 전부 사라진다. 남길 것은 커밋·푸시한다.

## 3. 읽는 순서

| 순서 | 문서 | 무엇이 있나 |
|---|---|---|
| 1 | `docs/agent_memory/MEMORY.md` 와 각 메모리 | 사용자 규칙(seed 하나, 이분법 실험, 워크플로우 금지 등)과 최신 상태 요약 |
| 2 | `AGENTS.md` | 레포 작업 규칙(백그라운드 실행, `pkill` 금지, 실행 중 런 코드 불변, 커밋은 요청 시) |
| 3 | `docs/handoff_project_2026-10-28.md` | **주 인수인계서.** 결과 전부(1장, 1.5·1.6절이 최신), 함의, 논문 합의(3장), 기제(4장), 코드(5장), 남은 일(6장) |
| 4 | `docs/decisions.md` | 확정 설정과 근거. 바꾸려면 사용자에게 먼저 묻는다 |
| 5 | `README.md` | 명령(`scripts/run.sh`, `arc.small`, 평가) |
| 6 | `docs/research/stdp_wta_rule_synapse_2026-10-28/README.md` | STDP+WTA로 lasso 대체 조사: 대체 불가, 규칙 헤드 한 걸음 = LCA 한 걸음, 배치 영역은 고정점에서 멂 |
| 7 | `docs/handoff_toy_torus_search_2026-10-28.md` | 원환 탐색 토이(3요소 탐색 성공, 통합 실패) |
| 8 | `docs/design_rule_heads_2026-10-28.md` | 헤드별 규칙 시냅스 설계(시험해서 버림), 전이 헤드가 안 되는 이유 |
| 필요할 때 | `docs/blocks_and_backprop.md`, `docs/handoff_agent_2026-10-26.md`, `docs/handoff_user_2026-10-26.md`, `docs/evidence/` | 블록 수와 역전파(예전 결론), 10-26 인수인계, 런별 로그 |

## 4. 현재 상태 (한 화면)

- **마감:** Kaggle ARC-AGI-2 코드 제출 11-02(인터넷 없는 추론 노트북이 실제로 돌아야 함), 논문 11-08.
- **ARC-2:** 모든 평가 0/172. 원인은 가설 족(S) 위반으로 해석. 0%로 제출한다.
- **ARC-1 전체:** 예전 16/10(학습 2D RoPE) pass@2 0.1875, TRM 체크포인트 0.4175(우리 평가).
- **≤10 하네스(278쌍):** 30/10 1D + 출력 FFN + wd 행렬·λ가 20만 스텝 119로 아직 상승, 8/4는 9만 스텝부터 105~111 평탄.
  하지만 처음 보는 ARC-1 eval ≤10 88쌍에서는 30/10 38, 8/4 37, TRM 40으로 구분되지 않는다. 깊이 이득은 training 과제 쪽에만 있다.
- **10-28 밤에 시험해서 상한을 못 올린 것:** θ 걸음 1/λ_max(브랜치 `lmax`), 규칙 헤드 4개(브랜치 `heads`). 10/10은 7만 3천
  스텝에서 중단해 "전체 깊이 대 역전파 창"은 답이 없다.
- **준비만 된 것:** `scripts/capped_run.sh`로 ARC-1 전체 30/10 5k 에폭을 $50 상한으로 돌리는 런. 사용자 지시 전에는 띄우지 않는다.

## 5. 브랜치

- `main`: 모든 결과와 문서. 기본 코드는 여기.
- `lmax`: `lmax_step`, `ista_iters` 옵션(θ 근접 걸음 1/λ_max, 블록당 K걸음).
- `heads`: `rule_heads` 옵션(추가 규칙 시냅스), 테스트 3개.

## 6. 남은 결정 (사용자)

1. Kaggle 노트북을 언제 만들지(11-02 마감; 판 수를 시간 제한에 맞추고 가중치를 Kaggle Dataset으로).
2. ARC-1 전체 런을 돌릴지, 돌리면 30/10인지(근거: ≤10 278쌍) 아니면 더 싼 깊이인지(88쌍에서는 차이 없음).
3. 10/10을 끝까지(약 1시간) 다시 돌릴지.
4. 논문 결과표에 넣을 ≤10 비교의 표현(같은 하네스 약 24 대 약 119, 88쌍 38 대 40).
