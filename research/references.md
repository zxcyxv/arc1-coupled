# 참고문헌 후보 (2026-10-30)

논문(ARC Prize 2026 논문 트랙, 11-08 마감)의 참고문헌 바탕이다. 관련성으로 세 등급을 나눈다.

- **A 직접:** 비교 대상, 제출 벤치마크, 모델이 그대로 쓰는 구성 요소, 두 논지(재귀 위의 STDP, TEM의 구조·내용 분리)가 직접 기대는 문헌. 논문에 들어간다.
- **B 애매:** 논문 구성에 따라 넣을 수 있는 것. 다른 ARC 접근, 최근 재귀 모델, 시퀀스 축 메모리 모델, 한계 절에서만 쓸 신경과학.
- **C 거의 무관:** 조사 문서(`docs/research/stdp_wta_rule_synapse_2026-10-28/`) 안에서만 쓴 기술 문헌과 배경 지식.

"쓸 곳"은 `docs/handoff_project_2026-10-28.md` 3장의 구성을 따른다. 배경, 비교, 논지1(재귀 시간축과 STDP), 논지2(TEM), 기제(4장),
실험, 외삽(스도쿠), 한계(3.6절), 관련 연구.

"서지" 표기:
- **확인:** 출판사, 학회 proceedings, arXiv 목록의 검색 결과로 저자, 제목, 발표처를 맞췄다. 이 세션의 네트워크 정책 때문에 arxiv.org, PubMed,
  Crossref 페이지를 직접 열지는 못했다. 최종 bib을 만들기 전에 원문 페이지로 한 번 더 본다.
- **수정:** 기억과 달라서 고친 것. 무엇을 고쳤는지 적는다.
- **부분:** 서지나 내용 주장의 일부가 확인되지 않았다.
- **신규:** 이번 검색에서 새로 찾은 것(검색 결과 기준).
- **미확인:** 이번에 확인하지 않았다(조사 문서의 표기 그대로).

## A. 직접

### A.1 벤치마크와 대회

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| chollet2019 | Chollet (2019). On the Measure of Intelligence. arXiv:1911.01547 | ARC의 정의 | 배경 | 확인 |
| chollet2025arc2 | Chollet, Knoop, Kamradt, Landers, Pinkard (2025). ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems. arXiv:2505.11831 | 제출 트랙. 결과표의 0/172 | 배경, 실험 | 확인 |
| arcprize2025report | Chollet, Knoop, Kamradt, Landers (2026). ARC Prize 2025: Technical Report. arXiv:2601.10904 | 대회 맥락. 2025 논문상 1위가 TRM, 3위가 CompressARC. Kaggle ARC-AGI-2 비공개 최고 NVARC 24.03%, ARChitects 16.53%, MindsAI 12.64% | 배경 | 확인 |
| moskvichev2023 | Moskvichev, Odouard, Mitchell (2023). The ConceptARC Benchmark: Evaluating Understanding and Generalization in the ARC Domain. TMLR (08/2023). arXiv:2305.07141 | TRM 분할의 학습 과제 160개 | 실험(데이터) | 확인 |

### A.2 비교 대상: 재귀 추론 모델

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| jolicoeur2025trm | Jolicoeur-Martineau (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv:2510.04871 | 주 비교 대상. 코드 c011037, 데이터 빌더, 학습 레시피(배치 768, lr 1e-4, EMA, loops 16). 공개 체크포인트를 우리 평가로 pass@2 0.4175 | 비교, 실험 | 확인 |
| wang2025hrm | Wang, Li, Sun, Chen, Liu, Wu, et al. (2025). Hierarchical Reasoning Model. arXiv:2506.21734 | TRM의 전신, 재귀 transformer 계보. 관련 연구에서 TEM을 소개만 한다(논지2의 선례 구분) | 배경, 논지2 | 확인 |
| gao2025urm | Gao, Chen, Xiao, Xing, Tao, Luo, et al. (2025). Universal Reasoning Model. arXiv:2512.14693 | 증강 빌더(TRM/URM)와 RMSNorm 형식. 보고 수치 ARC-1 53.8%, ARC-2 16.0%(pass@1). 학습 곡선(Fig. 5)을 인용 | 비교 | 확인 |
| arcprize2025hrm | ARC Prize (2025-08-15). The Hidden Drivers of HRM's Performance on ARC-AGI. 블로그, arcprize.org/blog/hrm-analysis | 위계 구조의 몫은 작고, 바깥 반복 루프가 주된 동력이다. ACT를 끄고 위계를 강조하지 않는 근거 | 기제, 실험 | 부분: 필자 미확인 |
| mcgovern2025 | McGovern (2025). Test-time Adaptation of Tiny Recursive Models. arXiv:2511.02886 | 인수인계서 1.2절이 "TRM 재현(Trelis), Fig. 1"로 인용한 문헌. 실제로는 TRM을 공개 과제로 사전학습한 뒤 대회 과제로 미세조정하는 연구다. TRM은 처음 보는 과제에 재학습이 필요하고, 우리는 시범을 문맥으로 읽는다는 대비에도 쓴다 | 비교 | 수정: 재현 논문이 아님. Fig. 1 곡선은 원문에서 확인 |
| royeazar2025 | Roye-Azar, Vargas-Naranjo, Ghai, Balamurugan, Amir (2025). Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute. arXiv:2512.11847 | TRM 공개 체크포인트를 재분석했다. 퍼즐 ID를 비우거나 무작위로 하면 정확도가 0이 되고, 1000판 투표가 pass@1을 약 11 올린다. TRM에서 시범 → 질의 통로가 퍼즐 임베딩이라는 대비의 근거 | 비교, 논지2 | 신규 |
| dehghani2019ut | Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser (2019). Universal Transformers. ICLR. arXiv:1807.03819 | 가중치 공유 재귀 transformer의 원형 | 논지1 | 확인 |
| engdahl2026bdhcq | Engdahl, Kosowski, Chorowski, Stamirowska, Uznański, Jiang, et al. (2026). BDH-CQ: In-Context Learning with Recurrent Latent Reasoning. arXiv:2608.09888 | 150M, ARC-AGI-1 공개 eval pass@2 29.5%. 시범을 문맥으로 읽는 재귀 모델이라 구조가 가장 가까운 경쟁자다. 인수인계서는 "전면 attention"으로 적었다 | 관련 연구, 논지1(새로움의 경계) | 신규: 저자 순서는 paperswithcode 기준 |
| kosowski2025bdh | Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz (2025). The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507 | Hebbian 가소성 + attention 뇌 모델. 시간 비대칭 창과 재귀 깊이 축 기억이 있는지 본문에서 확인해 차이를 적어야 한다 | 논지1(새로움의 경계) | 확인 |

### A.3 논지1: Hebb, STDP, 형식화

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| hebb1949 | Hebb (1949). The Organization of Behavior: A Neuropsychological Theory. Wiley. p. 62 | 원문의 인과성: "When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, ..." | 논지1 첫 문장 | 부분: 인용문은 "excite **a** cell B". 쪽 번호는 2차 출처라 1949 Wiley판으로 인용 |
| markram1997 | Markram, Lübke, Frotscher, Sakmann (1997). Regulation of Synaptic Efficacy by Coincidence of Postsynaptic APs and EPSPs. Science 275(5297):213–215 | 스파이크 시각 의존 가소성의 실험 발견 | 논지1 | 확인 |
| bi1998 | Bi & Poo (1998). Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type. J Neurosci 18(24):10464–10472 | STDP 창의 측정 | 논지1 | 확인 |
| markram2011history | Markram, Gerstner, Sjöström (2011). A history of spike-timing-dependent plasticity. Front Synaptic Neurosci 3:4 | Hebb의 가설부터 STDP까지의 역사. 논지1 서술의 뼈대 | 논지1 | 신규 |
| caporale2008 | Caporale & Dan (2008). Spike timing-dependent plasticity: a Hebbian learning rule. Annu Rev Neurosci 31:25–46 | STDP의 표준 리뷰 | 논지1 | 신규 |
| gerstner2002math | Gerstner & Kistler (2002). Mathematical formulations of Hebbian learning. Biol Cybern 87(5–6):404–415 | 상관·공분산 형식의 Hebb와 스파이크 시각 형식을 나란히 정리한다. "형식화된 Hebb 대 시각 가설"과 창 적분 분해의 가장 안전한 출처 | 논지1 | 신규 |
| kempter1999 | Kempter, Gerstner, van Hemmen (1999). Hebbian learning and spiking neurons. Phys Rev E 59(4):4498–4514 | 창의 적분(직류 성분)이 rate Hebb 항이 된다는 분해. STDP는 Hebb의 일반화라는 주장 | 논지1 | 부분: 분해식을 본문에서 직접 보지 못했다. gerstner2002math와 함께 인용 |
| song2000 | Song, Miller, Abbott (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. Nat Neurosci 3(9):919–926 | 경쟁이 생기려면 창의 적분이 음수여야 한다. 학습된 창의 M₀가 음수인 것(8헤드 중 7)과 연결 | 논지1, 기제 | 확인 |
| xie2000 | Xie & Seung (2000). Spike-based Learning Rules and Stabilization of Persistent Neural Activity. NIPS 12, pp. 199–208 | 창의 1차 모멘트가 pre × d(post)/dt이므로 고정점에서 0이다. "홀수 성분은 궤적을 기억한다"의 수학적 근거 | 논지1 | 부분: 이들의 창은 anti-Hebbian이라 도함수 항의 부호가 표준 STDP와 반대다. "직류 성분 = Hebb"는 초록에 없다 |
| bengio2015stdp | Bengio, Mesnard, Fischer, Zhang, Wu (2015). STDP as presynaptic activity times rate of change of postsynaptic activity. arXiv:1509.05936 | Δw ∝ pre × d(post)/dt가 STDP를 재현한다. xie2000과 같은 근거 | 논지1 | 확인 |
| sejnowski1977 | Sejnowski (1977). Storing covariance with nonlinearly interacting neurons. J Math Biol 4(4):303–321 | 부호 있는 Hebbian(공분산 규칙)의 전례. 활동을 부호 있는 집단 부호로 보는 용어 정의 | 논지1, 용어 | 확인 |

### A.4 논지1: fast weight → linear attention 계보, 재귀 위의 가소성

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| schmidhuber1992 | Schmidhuber (1992). Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks. Neural Computation 4(1):131–139 | fast weight의 원형 | 논지1 | 확인 |
| ba2016 | Ba, Hinton, Mnih, Leibo, Ionescu (2016). Using Fast Weights to Attend to the Recent Past. NeurIPS 29. arXiv:1610.06258 | fast weight를 attention으로 쓰는 계보 | 논지1 | 확인 |
| katharopoulos2020 | Katharopoulos, Vyas, Pappas, Fleuret (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML, PMLR 119:5156–5165. arXiv:2006.16236 | bands 읽기(q̃ Σ v k̃ᵀ)의 기본형 | 논지1, 기제 | 확인 |
| schlag2021 | Schlag, Irie, Schmidhuber (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML, PMLR 139:9355–9366. arXiv:2102.11174 | 계보의 중심 연결고리(linear attention = fast weight, 델타 규칙) | 논지1 | 확인 |
| yang2024deltanet | Yang, Wang, Zhang, Shen, Kim (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS. arXiv:2406.06484 | λ = 0인 규칙 헤드는 DeltaNet형 델타 규칙이다 | 논지1, 기제 | 확인 |
| irie2025primer | Irie & Gershman (2025). Fast weight programming and linear transformers: from machine learning to neurobiology. arXiv:2508.08435 | fast weight, linear transformer, SSM을 시냅스 가소성 모델과 잇는 primer. 시간 비대칭이나 STDP를 다루면 논지1의 새로움과 겹치므로 본문 확인이 필요하다 | 논지1(새로움의 경계) | 신규 |
| orvieto2023lru | Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De (2023). Resurrecting Recurrent Neural Networks for Long Sequences. ICML, PMLR 202:26670–26698. arXiv:2303.06349 | bands 극점 매개화의 출처. z = exp(−e^ν + iθ), γ = √(1 − \|z\|²) | 기제 | 부분: LRU는 위상도 exp(θ_log)로 지수화하고, γ는 √(1 − \|λ\|²)로 초기화하는 별도 파라미터일 수 있다. "LRU식 매개화의 변형"으로 쓴다 |
| miconi2018 | Miconi, Clune, Stanley (2018). Differentiable Plasticity: Training Plastic Neural Networks with Backpropagation. ICML, PMLR 80. arXiv:1804.02464 | 역전파로 가소성 계수를 학습한다 | 논지1 | 확인 |
| miconi2019 | Miconi, Rawal, Clune, Stanley (2019). Backpropamine: Training Self-Modifying Neural Networks with Differentiable Neuromodulated Plasticity. ICLR. arXiv:2002.10585 | 신경조절 가소성. 조사 문서가 "가장 가까운 선행 연구"로 지목. 3요소 규칙과의 차이 | 논지1, 한계 | 확인 |
| fu2026tah | Fu, You, Chen, Dai, Yang, Wang (2026). Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning. ICML, PMLR 306. arXiv:2511.08577 | "duo-causal" attention: 토큰이 앞 위치와 더 얕은 반복 단계에 attend한다. 재귀 깊이 축 기억의 가장 가까운 사례(감쇠 흔적은 아님) | 논지1(새로움의 경계) | 신규 |

### A.5 규칙 헤드: lasso, ISTA, 맥락 내 회귀

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| tibshirani1996 | Tibshirani (1996). Regression Shrinkage and Selection via the Lasso. J R Stat Soc B 58(1):267–288 | lasso | 기제 | 확인 |
| daubechies2004 | Daubechies, Defrise, De Mol (2004). An iterative thresholding algorithm for linear inverse problems with a sparsity constraint. Comm Pure Appl Math 57(11):1413–1457. arXiv:math/0307152 | ISTA. 규칙 헤드 갱신 θ ← soft(θ − (θG − S), λ) | 기제 | 확인 |
| rozell2008 | Rozell, Johnson, Baraniuk, Olshausen (2008). Sparse Coding via Thresholding and Local Competition in Neural Circuits. Neural Comput 20(10):2526–2563 | LCA. 규칙 헤드 한 걸음 = LCA 한 걸음(신경 해석) | 기제, 논지1 | 확인 |
| vonoswald2023 | von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, et al. (2023). Transformers Learn In-Context by Gradient Descent. ICML, PMLR 202:35151–35174. arXiv:2212.07677 | 규칙 헤드 = "문맥 위의 델타 규칙"의 ML 선행 연구. 조사 문서가 "반드시 넣는다"고 적었다 | 기제, 관련 연구 | 확인 |
| akyurek2023 | Akyürek, Schuurmans, Andreas, Ma, Zhou (2023). What Learning Algorithm Is In-Context Learning? Investigations with Linear Models. ICLR. arXiv:2211.15661 | 맥락 내 학습이 ridge/GD를 구현한다. vonoswald2023과 같은 이유 | 기제, 관련 연구 | 확인 |
| wang2026ttr | Wang, Shi, Fox (2026). Test-time regression: a unifying framework for designing sequence models with associative memory. JMLR 27 (25-0903). arXiv:2501.12352 | 연합 기억 = 시험 시 회귀. 규칙 헤드(lasso 회귀)와 bands(선형 연합 기억)를 한 틀로 설명할 수 있다 | 기제 | 수정: JMLR 2026 게재 |
| bertinetto2019 | Bertinetto, Henriques, Torr, Vedaldi (2019). Meta-learning with Differentiable Closed-form Solvers. ICLR. arXiv:1805.08136 | R2-D2. 지원 집합에서 ridge 헤드를 닫힌 해로 풀고 그 해를 통해 역전파한다. 규칙 헤드의 가장 가까운 메타학습 선례(우리도 닫힌 해 ridge Θ를 시도했다) | 기제, 관련 연구 | 확인 |
| lee2019metaoptnet | Lee, Maji, Ravichandran, Soatto (2019). Meta-Learning with Differentiable Convex Optimization. CVPR. arXiv:1904.03758 | MetaOptNet. 볼록 최적화 헤드, 같은 계열 | 관련 연구 | 확인: 쪽 번호는 CVF와 IEEE가 다름 |
| donoho1994 | Donoho & Johnstone (1994). Ideal spatial adaptation by wavelet shrinkage. Biometrika 81(3):425–455 | universal threshold σ√(2 log n). λ 사전값 κ/√n의 근거 | 기제 | 확인 |

### A.6 논지2: TEM

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| whittington2020tem | Whittington, Muller, Mark, Chen, Barry, Burgess, Behrens (2020). The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation. Cell 183(5):1249–1263.e23 | 구조·내용 분리. φ(공유 구조 코드)와 θ(과제별 결합) | 논지2 | 확인 |
| whittington2022temt | Whittington, Warren, Behrens (2022). Relating Transformers to Models and Neural Representations of the Hippocampal Formation. ICLR. arXiv:2112.04035 | TEM-t. 프레임 항 = 위치 key → 색 value 인출 | 논지2 | 확인 |
| rambaud2025mapformer | Rambaud, Mascarenhas, Lakretz (2025). MapFormer: Self-Supervised Learning of Cognitive Maps with Input-Dependent Positional Embeddings. arXiv:2511.19279 | TEM-t를 잇는 가장 가까운 연구(입력 의존 위치 부호로 구조와 내용을 분리). "To our knowledge"의 근거 | 논지2 | 확인: v3(2026-02) 제목은 "...Positional Encoding". 프리프린트 |

### A.7 구성 요소와 학습 레시피

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| su2024rope | Su, Lu, Pan, Murtadha, Wen, Liu (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568:127063. arXiv:2104.09864 | RoPE(1D 고정, TRM 주파수) | 기제 | 확인 |
| prieto2025 | Prieto, Barsbey, Mediano, Birdal (2025). Grokking at the Edge of Numerical Stability. ICLR. arXiv:2501.04697 | StableMax. TRM lm 손실의 stablemax CE | 기제 | 확인 |
| everett2024 | Everett, Xiao, Wortsman, Alemi, Novak, Liu, et al. (2024). Scaling Exponents Across Parameterizations and Optimizers. ICML, PMLR 235:12666–12700. arXiv:2407.05872 | Adam-atan2(TRM이 쓰는 adam-atan2 0.0.3) | 기제 | 확인 |
| zhang2019rmsnorm | Zhang & Sennrich (2019). Root Mean Square Layer Normalization. NeurIPS. arXiv:1910.07467 | 블록의 post-RMSNorm | 기제 | 확인 |
| shazeer2020 | Shazeer (2020). GLU Variants Improve Transformer. arXiv:2002.05202 | SwiGLU FFN, 출력 FFN | 기제 | 확인 |
| vaswani2017 | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, et al. (2017). Attention Is All You Need. NIPS 30, pp. 5998–6008. arXiv:1706.03762 | 기본 | 기제 | 확인 |

### A.8 외삽 (스도쿠)

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| schwarzschild2021 | Schwarzschild, Borgnia, Gupta, Huang, Vishkin, Goldblum, Goldstein (2021). Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks. NeurIPS. arXiv:2106.04537 | 반복 수를 늘려 더 어려운 문제로 외삽 | 외삽 | 확인: NeurIPS는 2차 목록 기준 |
| bansal2022 | Bansal, Schwarzschild, Borgnia, Emam, Huang, Goldblum, Goldstein (2022). End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking. NeurIPS. arXiv:2202.05826 | "overthinking" 없는 외삽. 스도쿠 1만 블록 외삽(재발 0, 못 푼 문제가 떠돎)과 직접 비교 | 외삽 | 확인 |

## B. 애매

### B.1 다른 ARC 접근

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| akyurek2025ttt | Akyürek, Damani, Zweiger, Qiu, Guo, Pari, et al. (2025). The Surprising Effectiveness of Test-Time Training for Few-Shot Learning. ICML, PMLR 267:942–963. arXiv:2411.07279 | 시험 시 학습(8B LM, ARC 53.0%). 우리는 재학습 없이 시범을 문맥으로 읽는다는 대비 | 관련 연구 | 확인: v1 제목 "...for Abstract Reasoning" |
| franzen2025 | Franzen, Disselhoff, Hartmann (2025). Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective. ICML, PMLR 267:17657–17671. arXiv:2505.07859 | 증강 판 투표(우리 투표와 같은 발상). ARChitects | 관련 연구 | 확인 |
| li2025indtrans | Li, Hu, Larsen, Wu, Alford, Woo, et al. (2025). Combining Induction and Transduction for Abstract Reasoning. ICLR. arXiv:2411.02272 | 귀납 대 전이. 우리 θ는 귀납적 규칙이고 출력은 전이적이다 | 관련 연구 | 확인: ICLR판과 arXiv v4의 저자 목록이 다름 |
| liao2025compressarc | Liao & Gu (2025). ARC-AGI Without Pretraining. arXiv:2512.06104 | 사전학습 없이 과제마다 MDL 최소화(약 76K 파라미터, ARC-1 eval 약 20%). 2025 논문상 3위 | 관련 연구 | 수정: arXiv 번호 추가 |
| hu2025varc | Hu, Cy, Qiu, Ding, Wang, Zhu, et al. (2025). ARC Is a Vision Problem! arXiv:2511.14761 | VARC, 처음부터 학습해 ARC-1 60.4%. 캔버스·평행이동 증강 | 관련 연구 | 확인 |
| hodel2024rearc | Hodel (2024). Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation. arXiv:2404.07353 | re-arc 생성 데이터(E3 생성 데이터 계획은 취소) | 관련 연구 | 확인 |
| freinschlag2026 | Freinschlag, Bertram, Kobler, Mayr, Klambauer (2026). Symbol-Equivariant Recurrent Reasoning Models. arXiv:2603.02193 | 색·기호 등변 재귀 모델(약 2M). 우리는 색 등변이 아니어서 색 증강으로 근사한다(설계 근거) | 관련 연구, 기제 | 신규: 저자 순서가 출처마다 다름 |

### B.2 최근 재귀·루프 모델 (2025~2026)

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| huang2026eqr | Huang, Geng, Kolter (2026). Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning. ICML. arXiv:2605.21488 | 끌개를 학습하면 반복을 늘릴 수 있다. ≤10 loops 1의 깊이 프로필이 끌개가 아니라 언덕이었던 것, loops > 1 외삽 실험과 직접 연결 | 외삽, 한계 | 신규 |
| li2026infiloop | Li, Muhtar, He, Su, Yin, Liu (2026). Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals. arXiv:2610.11570 | InfiLoop(7M). 반복 수만 번 외삽, ARC-1 49.1%, ARC-2 13.6%(pass@2). 이들의 TRM 재학습은 ARC-2 4.6% | 외삽, 비교 | 신규: 제목·저자는 papers.cool 기준 |
| movahedi2026fprm | Movahedi, Milovanović, Feigin, Theus, Hofmann, Boeva, et al. (2026). Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers. ICML. arXiv:2606.18206 | pre-norm + residual 배율로 깊은 루프를 안정화. ARC-1 47.5%(pass@2, InfiLoop 인용 기준) | 관련 연구 | 신규: ICML 목록 제목은 "Fixed-Point Reasoning: ..." |
| suleymanzade2026 | Suleymanzade, Lee, Eijkelboom, Boffi, Ceylan, Kim (2026). Thinking with Looped Flows. arXiv:2609.11801 | TRM 구조 + 국소 denoising 목적. ARC-1 58.8%, ARC-2 12.2%(pass@2) | 관련 연구 | 신규 |
| sghaier2026ptrm | Sghaier, Parviz, Jolicoeur-Martineau (2026). Probabilistic Tiny Recursive Model. NeurIPS. arXiv:2605.19943 | 추론 시간만으로 TRM 체크포인트를 개선(스도쿠, 미로, ARC-2) | 관련 연구 | 신규: 저자 순서가 출처마다 다름 |
| asadulaev2025dis | Asadulaev, Banerjee, Karray, Takac (2025). Deep Improvement Supervision. arXiv:2511.16886 | TRM에 단계별 목표를 줘서 forward를 18배 줄이고 ARC-1 24% | 관련 연구 | 신규 |
| ge2025hrm | Ge, Liao, Poggio (2025). Hierarchical Reasoning Model: A Critical Supplementary Material. arXiv:2510.00355 | HRM 비판적 검토 | 관련 연구 | 신규: v2 제목 확인 필요 |
| ren2026 | Ren & Liu (2026). Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models. arXiv:2601.10679 | HRM의 기제 분석(증강으로 스도쿠 54.5 → 96.9%) | 관련 연구 | 신규 |
| neri2026 | Neri, Gaeta, Ceschini, Facchinetti, Fico (2026). Nested Shared-Operator Recursion on ARC-AGI-1: An Empirical Study of HRM, TRM and URM. arXiv:2610.26666 | 파라미터를 맞춘 중첩 재귀 변형은 일관된 이득이 없다. 평가는 16세그먼트, halt 끔(우리 평가와 같음) | 관련 연구 | 신규 |
| shu2026loopvit | Shu, Qiu, Zhu, Chen, Liu, Yang (2026). LoopViT: Scaling Visual ARC with Looped Transformers. arXiv:2602.02156 | 가중치 묶은 루프 ViT(18M), ARC-1 65.8% | 관련 연구 | 신규 |
| geiping2025 | Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, et al. (2025). Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. NeurIPS. arXiv:2502.05171 | 재귀 깊이로 시험 시 연산을 확장. 스도쿠 외삽의 맥락 | 외삽, 관련 연구 | 확인 |
| bae2025mor | Bae, Kim, Bayat, Kim, Ha, Schuster, et al. (2025). Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. NeurIPS. arXiv:2507.10524 | 재귀 단계별 KV 캐시 대 첫 단계 KV 재사용. 재귀 축 기억의 한 형태 | 논지1(새로움의 경계) | 신규 |
| graves2016act | Graves (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv:1603.08983 | ACT. 우리는 끄고 loops 16 고정 | 기제 | 확인 |
| bai2019deq | Bai, Kolter, Koltun (2019). Deep Equilibrium Models. NeurIPS. arXiv:1909.01377 | 고정점 반복과 암묵 기울기(`docs/blocks_and_backprop.md`의 역전파 창 논의) | 기제 | 확인 |
| giannou2023 | Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos (2023). Looped Transformers as Programmable Computers. ICML, PMLR 202. arXiv:2301.13196 | 루프 transformer의 표현력 | 관련 연구 | 확인 |
| yang2024looped | Yang, Lee, Nowak, Papailiopoulos (2024). Looped Transformers are Better at Learning Learning Algorithms. ICLR. arXiv:2311.12424 | 루프 transformer의 맥락 내 학습 | 관련 연구 | 확인 |
| palm2018rrn | Palm, Paquet, Winther (2018). Recurrent Relational Networks. NeurIPS, pp. 3368–3378. arXiv:1711.08028 | 스도쿠 재귀 기준선 | 외삽 | 확인 |
| wang2019satnet | Wang, Donti, Wilder, Kolter (2019). SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver. ICML, PMLR 97. arXiv:1905.12149 | 스도쿠 기준선 | 외삽 | 수정: 제목 전체 |
| power2022grokking | Power, Burda, Edwards, Babuschkin, Misra (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177 | 장기 학습 확인 런(100k 연장)의 맥락. 시험 CE가 올라도 정답 수가 오르는 현상 | 실험 | 확인 |

### B.3 시퀀스 축 메모리 모델 (bands와 형식이 닮음)

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| sun2023retnet | Sun, Dong, Huang, Ma, Xia, Xue, et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621 | 복소 회전 감쇠 linear attention. bands의 z_r와 형식이 닮았지만 축이 시퀀스다 | 관련 연구 | 확인 |
| gu2022s4 | Gu, Goel, Ré (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR. arXiv:2111.00396 | 대각 복소 SSM | 관련 연구 | 확인 |
| gu2024mamba | Gu & Dao (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. COLM. arXiv:2312.00752 | 선택적 SSM | 관련 연구 | 부분: COLM은 2차 출처만 |
| dao2024mamba2 | Dao & Gu (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML, PMLR 235:10041–10071. arXiv:2405.21060 | SSM과 linear attention의 쌍대성 | 관련 연구 | 확인 |
| yang2024gla | Yang, Wang, Shen, Panda, Kim (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML, PMLR 235:56501–56523. arXiv:2312.06635 | 게이트 linear attention | 관련 연구 | 확인 |
| yang2025gdn | Yang, Kautz, Hatamizadeh (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR. arXiv:2412.06464 | 감쇠 + 델타 규칙 | 관련 연구 | 확인 |
| behrouz2025titans | Behrouz, Zhong, Mirrokni (2025). Titans: Learning to Memorize at Test Time. NeurIPS. arXiv:2501.00663 | 시험 시 기억 학습 | 관련 연구 | 수정: NeurIPS 2025 |
| sun2025ttt | Sun, Li, Dalal, Xu, Vikram, Zhang, et al. (2025). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. ICML, PMLR 267:57503–57522. arXiv:2407.04620 | 은닉 상태를 학습기로 둔다 | 관련 연구 | 수정: ICML 2025 |
| irie2021rfwp | Irie, Schlag, Csordás, Schmidhuber (2021). Going Beyond Linear Transformers with Recurrent Fast Weight Programmers. NeurIPS. arXiv:2106.06295 | 재귀 fast weight programmer | 논지1 | 확인 |
| ramsauer2021 | Ramsauer, Schäfl, Lehner, Seidl, Widrich, Adler, et al. (2021). Hopfield Networks is All You Need. ICLR. arXiv:2008.02217 | modern Hopfield = attention | 관련 연구 | 확인 |
| hopfield1982 | Hopfield (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS 79(8):2554–2558 | 연합 기억의 고전 | 논지1 | 확인 |
| hinton1987 | Hinton & Plaut (1987). Using Fast Weights to Deblur Old Memories. Proc. 9th Annual Conf. Cognitive Science Society, pp. 177–186 | "fast weight" 용어의 기원 | 논지1 | 확인 |
| garg2022 | Garg, Tsipras, Liang, Valiant (2022). What Can Transformers Learn In-Context? A Case Study of Simple Function Classes. NeurIPS. arXiv:2208.01066 | 맥락 내 함수 학습 | 관련 연구 | 확인 |
| qiao2025 | Qiao (2025). Understanding Transformers through the Lens of Pavlovian Conditioning. arXiv:2508.08289 | linear attention = Hebbian contiguity, DeltaNet = Rescorla–Wagner, 감쇠 변형 = stimulus trace. 논지1의 계보 서술과 겹치므로 본문 확인이 필요하다 | 논지1(새로움의 경계) | 신규: 이후 "Attention as Conditioning: ..."으로 제목 변경 |
| mondal2025 | Mondal & Kumar (2025). Attention via Synaptic Plasticity is All You Need: A Biologically Inspired Spiking Neuromorphic Transformer. arXiv:2511.14691 | STDP로 attention을 계산하는 스파이킹 transformer(forward 안에서만) | 관련 연구 | 신규 |
| chaudhary2025 | Chaudhary (2025). Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity. arXiv:2510.21908 | 디코더 transformer에 Hebbian/기울기 fast weight를 더함 | 관련 연구 | 신규 |
| heo2024 | Heo, Park, Han, Yun (2024). Rotary Position Embedding for Vision Transformer. ECCV. arXiv:2403.13298 | RoPE-Mixed(학습 2D 주파수). 학습 2D RoPE 기준선(R0)의 선례. RoPE 발견은 Theory에서 뺐으므로 실험 절에서만 쓴다 | 실험 | 확인 |

### B.4 메타학습, 최적화, 희소 부호화

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| finn2017maml | Finn, Abbeel, Levine (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML, PMLR 70:1126–1135. arXiv:1703.03400 | 메타학습의 기준 | 관련 연구 | 확인 |
| amos2017optnet | Amos & Kolter (2017). OptNet: Differentiable Optimization as a Layer in Neural Networks. ICML, PMLR 70:136–145. arXiv:1703.00443 | 미분 가능한 최적화 층 | 관련 연구 | 확인 |
| najarro2020 | Najarro & Risi (2020). Meta-Learning through Hebbian Plasticity in Random Networks. NeurIPS. arXiv:2007.02686 | 메타학습된 Hebbian 규칙 | 논지1 | 확인 |
| gregor2010lista | Gregor & LeCun (2010). Learning Fast Approximations of Sparse Coding. ICML, pp. 399–406 | LISTA. 학습되는 펼친 ISTA | 기제 | 확인 |
| chen2018lista | Chen, Liu, Wang, Yin (2018). Theoretical Linear Convergence of Unfolded ISTA and Its Practical Weights and Thresholds. NeurIPS, pp. 9061–9071. arXiv:1808.10038 | LISTA의 수렴 | 기제 | 확인 |
| beck2009fista | Beck & Teboulle (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM J Imaging Sci 2(1):183–202 | FISTA. θ 수렴 가속 후보(6장) | 한계 | 확인 |
| landweber1951 | Landweber (1951). An iteration formula for Fredholm integral equations of the first kind. Am J Math 73(3):615–624 | 유한 T 항등식(조사 문서 P3) | 기제 | 확인 |
| yao2007 | Yao, Rosasco, Caponnetto (2007). On Early Stopping in Gradient Descent Learning. Constr Approx 26(2):289–315 | 조기 정지 = 정규화(블록마다 한 걸음의 스펙트럼 필터 해석) | 기제 | 확인 |
| olshausen1996 | Olshausen & Field (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature 381(6583):607–609 | 희소 부호화의 원류(LCA의 배경) | 기제 | 확인 |

### B.5 신경과학 (논지1 보강, 한계 절)

| 키 | 문헌 | 이 프로젝트와의 관계 | 쓸 곳 | 서지 |
|---|---|---|---|---|
| gerstner1996 | Gerstner, Kempter, van Hemmen, Wagner (1996). A neuronal learning rule for sub-millisecond temporal coding. Nature 383(6595):76–78 | 실험보다 먼저 이론으로 제안한 비대칭 학습 창 | 논지1 | 확인 |
| abbott2000 | Abbott & Nelson (2000). Synaptic plasticity: taming the beast. Nat Neurosci 3(Suppl):1178–1183 | 가소성 리뷰 | 논지1 | 확인 |
| sejnowski1999 | Sejnowski (1999). The book of Hebb. Neuron 24(4):773–776 | 50주년 재평가. "Hebb의 원문은 인과적"의 보조 근거 | 논지1 | 신규 |
| levy1983 | Levy & Steward (1983). Temporal contiguity requirements for long-term associative potentiation/depression in the hippocampus. Neuroscience 8(4):791–797 | STDP 이전에 순서가 중요하다는 초기 증거 | 논지1 | 신규 |
| rao2001 | Rao & Sejnowski (2001). Spike-timing-dependent Hebbian plasticity as temporal difference learning. Neural Comput 13:2221–2237 | STDP = 시간차(도함수) 규칙 | 논지1 | 신규 |
| brown1990 | Brown, Kairiss, Keenan (1990). Hebbian synapses: biophysical mechanisms and algorithms. Annu Rev Neurosci 13 | Hebbian 시냅스 개념의 변천 | 논지1 | 신규: 쪽 475–511 미확인 |
| sompolinsky1986 | Sompolinsky & Kanter (1986). Temporal Association in Asymmetric Neural Networks. PRL 57(22):2861–2864 | 비대칭 연결로 순서를 기억하는 고전. 이론 절에서 쓰면 A | 논지1 | 확인 |
| fiete2010 | Fiete, Senn, Wang, Hahnloser (2010). Spike-time-dependent plasticity and heterosynaptic competition organize networks to produce long scale-free sequences of neural activity. Neuron 65(4):563–576 | STDP + 경쟁 → 순서열 | 논지1 | 확인 |
| luz2014 | Luz & Shamir (2014). The effect of STDP temporal kernel structure on the learning dynamics of single excitatory and inhibitory synapses. PLoS ONE 9(7):e101109 | 창 모양이 학습 동역학을 바꾼다(창 분석의 배경) | 기제 | 확인 |
| fremaux2016 | Frémaux & Gerstner (2016). Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules. Front Neural Circuits 9:85 | 3요소 규칙 | 한계 | 확인 |
| gerstner2018 | Gerstner, Lehmann, Liakoni, Corneil, Brea (2018). Eligibility Traces and Plasticity on Behavioral Time Scales: Experimental Support of NeoHebbian Three-Factor Learning Rules. Front Neural Circuits 12:53 | eligibility trace × 신경조절 | 한계 | 확인 |
| izhikevich2007 | Izhikevich (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. Cereb Cortex 17(10):2443–2452 | STDP × 도파민. 원환 토이 dopamine 조건의 원형 | 한계 | 확인 |
| bittner2017 | Bittner, Milstein, Grienberger, Romani, Magee (2017). Behavioral time scale synaptic plasticity underlies CA1 place fields. Science 357:1033–1036 | BTSP(초 단위 창) | 한계 | 확인 |
| bellec2020 | Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, et al. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. Nat Commun 11:3625 | e-prop. 재귀망의 eligibility trace 학습 | 한계 | 확인 |
| masquelier2007 | Masquelier & Thorpe (2007). Unsupervised learning of visual features through spike timing dependent plasticity. PLoS Comput Biol 3(2):e31 | STDP + WTA(조사 문서) | 한계 | 확인 |
| nessler2013 | Nessler, Pfeiffer, Buesing, Maass (2013). Bayesian computation emerges in generic cortical microcircuits through spike-timing-dependent plasticity. PLoS Comput Biol 9(4):e1003037 | STDP + WTA = 베이즈 계산. depression은 soft threshold와 같지 않다(조사 문서) | 한계 | 확인 |
| kappel2014 | Kappel, Nessler, Maass (2014). STDP installs in winner-take-all circuits an online approximation to hidden Markov model learning. PLoS Comput Biol 10(3):e1003511 | WTA + 측면 STDP로 전이 학습. 전이 헤드 논의 | 한계 | 확인 |
| behrens2018 | Behrens, Muller, Whittington, Mark, Baram, Stachenfeld, Kurth-Nelson (2018). What Is a Cognitive Map? Organizing Knowledge for Flexible Behavior. Neuron 100(2):490–509 | 구조 지식의 일반화(TEM의 배경) | 논지2 | 확인 |
| whittington2025 | Whittington, Dorrell, Behrens, Ganguli, El-Gaby (2025). A tale of two algorithms: Structured slots explain prefrontal sequence memory and are unified with hippocampal cognitive maps. Neuron 113(2):321–333.e6 | TEM 계열. 활동 슬롯(작업 기억) 대 시냅스(일화 기억). MapFormer가 기대는 논문 | 논지2 | 신규: 성만 확인 |
| altabaa2024 | Altabaa, Webb, Cohen, Lafferty (2024). Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers. ICLR. arXiv:2304.00195 | 관계 정보와 대상 특징을 분리(구조·내용 분리와 가까움) | 논지2 | 신규 |
| altabaa2025 | Altabaa & Lafferty (2025). Disentangling and Integrating Relational and Sensory Information in Transformer Architectures. ICML, PMLR 267:1271–1297. arXiv:2405.16727 | 감각 attention과 관계 attention을 나란히 | 논지2 | 신규 |
| liu2026modality | Liu & Shang (2026). Reasoning is a Modality. arXiv:2601.13562 | ARC에서 전역 제어 토큰과 격자 작업 토큰을 나눠 규칙을 반복 실행 | 관련 연구 | 신규: ARC-1 62.6%는 검색 요약 수치 |
| tik2018 | Tik, Sladky, Di Bernardi Luft, Willinger, Hoffmann, Banissy, et al. (2018). Ultra-high-field fMRI insights on insight: Neural correlates of the Aha!-moment. Hum Brain Mapp 39(8):3241–3252 | 7T fMRI. 통찰 순간에 VTA, 측좌핵, 미상핵이 활성(보상 회로). 유레카 = 내부 도파민의 근거 | 한계 | 확인 |
| oh2020 | Oh, Chesebrough, Erickson, Zhang, Kounios (2020). An insight-related neural reward signal. NeuroImage 214:116757 | 통찰 직후 전전두 감마 폭발(안와전두로 일부 위치 추정) | 한계 | 확인: EEG 연구이고 선조체 fMRI가 아님 |
| salvi2016 | Salvi, Bricolo, Kounios, Bowden, Beeman (2016). Insight solutions are correct more often than analytic solutions. Thinking & Reasoning 22(4):443–460 | 통찰 해가 분석 해보다 더 자주 맞다 | 한계 | 수정: 제목은 "Insight solutions ..." |
| kounios2014 | Kounios & Beeman (2014). The Cognitive Neuroscience of Insight. Annu Rev Psychol 65:71–93 | 통찰 연구 리뷰 | 한계 | 확인 |

## C. 거의 무관

| 키 | 문헌 | 비고 | 서지 |
|---|---|---|---|
| pourcel2025soar | Pourcel, Colas, Oudeyer (2025). Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI. ICML, PMLR 267:49659–49688. arXiv:2507.14172 | 2025 논문상 2위. LLM 진화적 프로그램 합성 | 신규 |
| arcprize2025runnersup | 2025 논문상 runner-up: Joffe & Eliasmith, Vector Symbolic Algebras for the ARC; Guichard, Reimers, Kvalsund, Lepperød, Nichele, ARC-NCA (arXiv:2505.08778); Ho et al., ArcMemo; Berman, From Parrots to Von Neumanns; Pang, Efficient Evolutionary Program Synthesis | 대회 맥락. 방법이 다름(기호 연산, NCA, LLM) | 신규: 제목만 |
| nvarc2025 | Sorokin & Puget, NVARC solution to ARC-AGI-2 2025 | Kaggle ARC-AGI-2 1위(24.03%). 제출 맥락 | 신규: 서지 미확인 |
| greenblatt2024 | Greenblatt (2024-06-17). Getting 50% (SoTA) on ARC-AGI with GPT-4o. Redwood Research blog | LLM 프로그램 표본추출 | 확인 |
| boberirizar2024 | Bober-Irizar & Banerjee (2024). Neural networks for abstraction and reasoning: Towards broad generalization in machines. Sci Rep 14. arXiv:2402.03507 | ARC 신경망 접근 개관 | 부분: 학술지 정보는 집계 사이트만 |
| wang2026trmmamba | Wang & Reid (2026). Tiny Recursive Reasoning with Mamba-2 Attention Hybrid. ICLR 워크숍. arXiv:2602.12078 | TRM 변형 | 신규: 검색에서만 봄 |
| stachenfeld2017 | Stachenfeld, Botvinick, Gershman (2017). The hippocampus as a predictive map. Nat Neurosci 20(11):1643–1653 | 해마 계승자 표현 | 확인 |
| george2021cscg | George, Rikhye, Gothoskar, Guntupalli, Dedieu, Lázaro-Gredilla (2021). Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps. Nat Commun 12:2392 | 인지 지도의 다른 모형 | 부분: 논문 번호 미확인 |
| wagner2004 | Wagner, Gais, Haider, Verleger, Born (2004). Sleep inspires insight. Nature 427(6972):352–355 | 수면과 통찰 | 확인 |
| stent1973 | Stent (1973). A physiological mechanism for Hebb's postulate of learning. PNAS 70(4):997–1001 | Hebb 가설의 기제 해석 | 신규 |
| diehl2015 | Diehl & Cook (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Front Comput Neurosci 9:99 | STDP 분류기 | 확인 |
| foldiak1990 | Földiák (1990). Forming sparse representations by local anti-Hebbian learning. Biol Cybern 64(2):165–170 | anti-Hebbian 탈상관 | 확인 |
| zylberberg2011 | Zylberberg, Murphy, DeWeese (2011). A Sparse Coding Model with Synaptically Local Plasticity and Spiking Neurons Can Account for the Diverse Shapes of V1 Simple Cell Receptive Fields. PLoS Comput Biol 7(10):e1002250 | SAILnet | 확인 |
| blumensath2009 | Blumensath & Davies (2009). Iterative hard thresholding for compressed sensing. Appl Comput Harmon Anal 27(3):265–274. arXiv:0805.0510 | IHT(L0 대 L1 비교) | 확인 |
| donoho2003 | Donoho & Elad (2003). Optimally sparse representation in general (nonorthogonal) dictionaries via ℓ1 minimization. PNAS 100(5):2197–2202 | 희소 복원 조건 | 확인 |
| martins2016 | Martins & Astudillo (2016). From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. ICML, PMLR 48:1614–1623. arXiv:1602.02068 | sparsemax | 수정: 부제 추가 |
| bengio2015bio | Bengio, Lee, Bornschein, Mesnard, Lin (2015). Towards Biologically Plausible Deep Learning. arXiv:1502.04156 | 생물학적 학습 | 확인: v1은 저자 4명 |
| scellier2017 | Scellier & Bengio (2017). Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation. Front Comput Neurosci 11:24 | 평형 전파 | 확인 |
| lillicrap2020 | Lillicrap, Santoro, Marris, Akerman, Hinton (2020). Backpropagation and the brain. Nat Rev Neurosci 21(6):335–346 | 역전파와 뇌 | 확인 |
| research-doc-only | Makhzani & Frey (arXiv:1312.5663); Balavoine, Romberg & Rozell 2012 (IEEE TNNLS 23(9)); Maass 2000 (Neural Comput 12); Legenstein, Naeger & Maass 2005 (Neural Comput 17(11)); Frémaux, Sprekeler & Gerstner 2010 (J Neurosci 30); Mozafari et al. 2018 (IEEE TNNLS); Polyak 1964; Fuchs 2004; Zhao & Yu 2006; Wainwright 2009; Engl, Hanke & Neubauer 1996; Donoho & Huo 2001; Gribonval & Nielsen 2003; Cohen, Dahmen & DeVore 2009; arXiv:2304.05526 | 조사 문서의 증명·대조에만 쓴 것 | 미확인 |

## 논문 쓰기 전에 확인할 것

1. **arXiv:2511.02886.** 인수인계서 1.2절과 메모리는 "TRM 재현(Trelis), Fig. 1"로 적었지만, 실제로는 McGovern의 "Test-time Adaptation of Tiny
   Recursive Models"다. Fig. 1이 인용한 곡선(1000판 pass@2 약 3% @ 75k → 10.5% @ 650k)인지 원문에서 보고 표현을 고친다.
2. **새로움의 경계.** 다음 문헌을 본문까지 읽고, 재귀 깊이 축 위의 학습된 시간 비대칭 창이 없음을 확인해 차이를 한 줄씩 적는다.
   - 가장 가까운 것: BDH와 BDH-CQ(ARC-1 29.5%), Think-at-Hard(반복 단계에 attend)
   - 계보 서술이 겹칠 수 있는 것: Irie & Gershman 2025, Qiao 2025
   - 그 밖에: Mixture-of-Recursions, Mondal & Kumar 2025

   검색 수준에서는 재귀 깊이 축에 감쇠 흔적을 두는 2024~2026 논문을 찾지 못했다.
3. **bands 매개화.** LRU와 같다고 쓰지 않는다. LRU는 위상을 지수화하고 γ를 따로 학습할 수 있다. "LRU식 극점 매개화, γ를 |z|에 묶은 변형"으로 쓴다.
4. **STDP 분해의 출처.**
   - 직류 성분 = Hebb: Kempter 1999의 식을 원문에서 보거나 Gerstner & Kistler 2002를 함께 인용한다.
   - 1차 모멘트 = pre × d(post)/dt: Xie & Seung 2000과 Bengio et al. 2015로 인용한다. Xie & Seung의 창은 anti-Hebbian이라는 점을 적는다.
5. **Hebb 인용문.** "excite a cell B", 1949 Wiley판 p. 62. 쪽은 원본에서 확인한다.
6. **통찰 문헌.** Oh 2020은 EEG(전전두 감마)이고, 보상 회로 fMRI 근거는 Tik 2018이다. Salvi 2016의 제목은 "Insight solutions ...".
7. **수치 비교.** pass@1(URM)과 pass@2(TRM, Looped Flows, InfiLoop), 공개 eval과 semi-private가 섞여 있다. 표에 넣을 때 지표를 맞춘다.
8. **TRM의 시범 통로.** Roye-Azar et al. 2025(퍼즐 ID를 비우면 0)와 McGovern 2025(대회 과제에 미세조정)는 "TRM은 시범을 문맥으로 읽지
   않는다"는 대비의 근거다. 논지2와 Kaggle 제출 서술(재학습 없이 숨겨진 과제의 시범을 읽음)에 쓴다.
9. **loops > 1 실험과 직접 이어지는 문헌.** Equilibrium Reasoners(끌개 학습)와 InfiLoop(수만 번 반복). 외삽 절을 쓸 때 함께 본다.
10. **발표처 갱신.** 기억 이후 바뀐 것: test-time regression은 JMLR 2026, TTT는 ICML 2025, Titans는 NeurIPS 2025, Geiping은 NeurIPS 2025.
    Mamba의 COLM 2024는 2차 출처만 있다.
