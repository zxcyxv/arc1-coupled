# 이 레포에서 일하는 규칙

- 결정 사항은 [docs/decisions.md](docs/decisions.md)를 따른다. 바꾸려면 사용자에게 먼저 묻고, 바꾸면 그 문서를 고친다.
- 어텐션(여기서는 bands 읽기)과 FFN 각각에서 residual을 더한 직후 RMSNorm을 둔다: `u = RMSNorm(h + read)`,
  `h = RMSNorm(u + FFN(u))`. RMSNorm은 토큰별 전체 채널의 FP32 제곱 평균, `x * rsqrt(mean(x**2) + 1e-5)`, 학습 파라미터 없음.
- 실행(학습, 벤치, 평가, 긴 테스트)은 기본으로 백그라운드에서 돌린다. 런은 `scripts/run.sh start`(setsid nohup)로 띄우고,
  감시는 실제 python 워커 PID로 한다. `pkill -f`는 쓰지 않는다.
- 실행 중인 런의 코드와 설정은 바꾸지 않는다. 재개는 같은 `--out`으로 다시 시작하면 되고, 모델 설정이나 데이터가 다르면 거부한다.
- 체크포인트는 레인 교체 시점에만 쓴다(carry 없음). 전체 체크포인트는 최근 `keep_last`개, EMA milestone은 vote 평가(5천 에폭)마다 남긴다. 디스크가 작다.
- 과적합이나 학습 진행은 같은 EMA 가중치, 같은 평가 조건으로 잰다. 학습 로그의 train exact는 16세그먼트를 마친 레인 기준이다.
- 장기 학습 중에는 quick 평가마다 보고하지 않는다. vote 평가(5천 에폭), 오류, 요청받은 시점에만 보고한다.
- 커밋과 푸시는 사용자가 요청할 때 한다.
