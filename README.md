# arc1-coupled

coupled 모델(bands 지각 + 과제별 lasso 규칙 헤드)을 ARC-AGI-1에서 TRM과 같은 조건으로 학습하는 전용 레포다.
결정 사항과 근거는 [docs/decisions.md](docs/decisions.md)에 있고, 내일 할 일의 순서와 대응 체크리스트는 [docs/handoff_2026-10-26.md](docs/handoff_2026-10-26.md)에 있다.
모델은 `zxcyxv/something` 커밋 `093911e`의 `lt/coupled.py`와 bands 경로를 옮긴 것이고, 같은 seed에서 초기 가중치와 출력이 같다.

## 인스턴스에서 (4×H100 SXM, on-demand)

```bash
git clone <this repo> && cd arc1-coupled
bash scripts/run.sh setup      # torch 2.11 cu128 설치(없으면), GPU, NCCL(P2P 실패 시 자동 우회), 디스크, 데이터 체크섬, 테스트
bash scripts/run.sh bench      # grad_accum 1/2/4의 s/step과 피크 메모리 -> runs/bench.json, 예상 일수
bash scripts/run.sh start      # runs/arc1 학습을 백그라운드로 시작(같은 명령으로 재개)
bash scripts/run.sh status     # 진행, s/step, ETA, 평가 결과, GPU 상태, 디스크
bash scripts/run.sh stop       # 다음 레인 교체(16스텝 이내)에서 체크포인트를 쓰고 종료
CKPT=runs/arc1/step_N.pt bash scripts/run.sh eval     # 1001판 투표 평가(백그라운드)
bash scripts/run.sh archive    # 로그와 평가를 docs/evidence/run_arc1로(체크포인트 제외)
```

- ARC-AGI-2는 `RUN=arc2 CONFIG=configs/arc2.json bash scripts/run.sh start`로 띄운다(status, stop, eval도 같은 `RUN`).
- 학습 길이는 50k 에폭이다. quick 평가가 포화하면 `run.sh stop`으로 멈추고, `CKPT=runs/<run>/step_N.pt bash scripts/run.sh eval`로 최종 평가를 돌린다.
- 설정은 `EXTRA`로 바꾼다. 예: `EXTRA="epochs=30000"`. 시간 제한은 `MAX_HOURS=72`.
- 학습 중 평가:
  - 1천 에폭마다 원본 판 1개(`quick`, 약 15초)
  - 5천 에폭마다 8판 투표(`vote`, ARC-1 약 1분). 이때 EMA 가중치 milestone(15 MB)을 저장한다
  - 끝나면 1001판 투표(`final`, ARC-1 약 3시간)와 제출 파일 `submission_step_N.json`
  - 결과는 `runs/arc1/eval.jsonl`에 쌓인다.
- `archive`는 milestone을 `checkpoints/run_<런>/`으로 복사한다. `git add -f`로 올려 두면 학습 뒤 아무 milestone이나 `CKPT=... bash scripts/run.sh eval`로 1001판 평가를 할 수 있다.
- 평가와 체크포인트의 종류는 [docs/decisions.md](docs/decisions.md)의 표에 있다.

## 구성

```text
arc/model.py         coupled 모델: bands 블록(흔적, W, 2D RoPE), lasso 규칙 헤드 Θ(+ 틀 색 Γ), 세그먼트 단위 carry
arc/data.py          TRM 빌더의 난수 흐름 재생(증강판, 평행이동) -> 압축 저장, 레인 구성, 학습 데이터 흐름
arc/train.py         DDP 학습: 동시 레인 교체, TRM 손실, AdamATan2, EMA, 레인 교체 시점 체크포인트(정확한 재개), 평가 일정
arc/evaluate.py      TRM 프로토콜 평가: 판별 예측 -> crop -> 역변환 -> 투표(득표, 시범 정합도) -> pass@k(과제 가중)
configs/arc1.json    본 런 설정
scripts/run.sh       setup / bench / start / status / stop / eval / archive
scripts/prepare_data.py   data/kaggle -> data/arc1_trm.npz (--verify로 TRM 원본 배열과 바이트 대조)
data/kaggle/         TRM이 쓰는 ARC-AGI-1 training, evaluation, ConceptARC JSON (TinyRecursiveModels kaggle/combined)
data/arc1_trm.npz    압축 저장본 7.9 MiB: 960과제, 판 876,405개, 학습 쌍 3,988, 시험 쌍 419
data/arc2_trm.npz    ARC-AGI-2 저장본: 1,280과제(training2 + evaluation2 + concept), 학습 쌍 5,573, 시험 쌍 172
tests/               모델 환원 성질, TRM 데이터 일치, 투표, 정확한 재개, DDP = 단일 프로세스
```

## 데이터가 TRM과 같다는 것

`scripts/prepare_data.py --verify DIR`은 TRM의 `dataset/build_arc_dataset.py`가 만든 배열 전부를 이 저장본으로 다시 만들어 바이트 단위로 대조한다. 2026-10-25 결과는 둘 다 불일치 0이었다.

| | 학습 캔버스 | 시험 캔버스 | mean_puzzle_examples | 판 |
|---|---:|---:|---:|---:|
| ARC-1 | 3,633,410 | 385,815 | 4.1458115825 | 876,405 |
| ARC-2 | 5,163,277 | 170,393 | 4.3325932322 | 1,191,729 |

`tests/fixtures/trm_sample*.npz`는 각 TRM 배열에서 뽑은 256개 표본이고, 테스트가 이를 다시 확인한다.

## 디스크

| 항목 | 크기 |
|---|---|
| torch를 설치한 venv | 약 7 GB |
| 이 레포 | 약 15 MB |
| 런 하나 | 약 0.5 GB (체크포인트 2개 × 약 60 MB, milestone 약 11개 × 15 MB, train.jsonl 약 80 MB) |
| torch.compile 캐시(/tmp) | 약 1 GB |

carry는 체크포인트에 넣지 않는다. 체크포인트를 레인 교체 시점에만 쓰기 때문이다.
