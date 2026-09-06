# cluster/ — CSI Lab GPU 클러스터 잡

이 사이트는 OpenOnDemand 의 터미널/SSH 가 막혀 있다. 로그인 노드 셸이 없으므로
**모든 것이 잡이다** — git clone 조차 잡으로 돈다 (`git_sync.sbatch`).

## 규칙 — 무엇을 던지는가

> **`cluster/` 바로 아래의 `*.sbatch` 만 던진다. 그 외에는 아무것도 던지지 않는다.**

OOD Job Composer → New Job → 파일 내용을 붙여넣고 Submit. `#SBATCH` 헤더가 파일 안에
다 들어 있으므로 추가 인자는 필요 없다.

| 던지지 않는 것 | 이유 |
|---|---|
| `*_body.sh` | `#SBATCH` 가 없다. wrapper 가 `source` 하는 공유 본체. 던지면 자원 요청 없이 돌다 죽는다 |
| `*.py` | 다른 스크립트가 컨테이너 안에서 실행하는 헬퍼 |
| `recompute_delta_stats.sh` | 로컬(dev 박스)에서 돌리는 데이터셋 준비 도구 |
| `Dockerfile` | `docker build` 용. 클러스터에서 쓰지 않는다 |

## 실험별 — 무엇을 어떤 순서로

E 와 F 는 **0번을 먼저 한 번** 돌려야 한다. `$HOME/lerobot` 체크아웃이 없으면 학습 잡이
`FATAL: ... _body.sh not found` 로 즉사한다. A~D 는 `main_job*.sbatch` 가 클론까지 하므로
0번이 필요 없다.

| # | 파일 | 언제 |
|---|---|---|
| 0 | `git_sync.sbatch` | `$HOME/lerobot` 최초 클론. 이후엔 학습 잡이 스스로 `git pull` 한다 |

### A. SmolVLA 300 epoch — SCRAPE `*_cap_10fps` 4종 (absolute action)

| # | 파일 | 비고 |
|---|---|---|
| 1 | **`main_job.sbatch`** | **이것 하나만 던진다.** 0번(git_sync)도 필요 없다 |

`main_job.sbatch` 가 순서대로 다 한다 — 토큰 확인 → `$HOME/lerobot` clone/pull →
데이터셋 4종 prefetch → 학습 체인 제출. GPU 를 요청하지 않고, 다운로드 동안만 작은
할당을 잡았다가 학습 잡을 큐에 넣고 끝난다.

```
GPU A (array 0):  push_cube  ──afterok──▶  extract_cube
GPU B (array 1):  close_box  ──afterok──▶  open_box
```

각 링크가 자기 `afterok` 에 매단 다음 링크를 즉시 큐잉한다. 앞 링크가 실패하거나
walltime 에 걸리면 SLURM 이 뒷 링크를 스스로 취소한다(`DependencyNeverSatisfied`).

네 단계 모두 멱등이라 실패 후 **그냥 다시 던지면 된다** — git 은 fast-forward, prefetch 는
Hub 와 일치하는 데이터셋을 건너뛰고, 제출 단계는 cap300 잡이 이미 큐에 있으면 거부한다.

`train_cap300.sbatch` / `train_cap300_body.sh` 는 `main_job` 이 알아서 쓴다. 손대지 않는다.

### B. SmolVLA — SCRAPE ablation_study Phase-1 (A1 · A2)

| # | 파일 | 비고 |
|---|---|---|
| 1 | **`main_job_phase1.sbatch`** | **이것 하나만 던진다.** 0번(git_sync)도 필요 없다 |

`main_job.sbatch` 와 같은 구조다 — 토큰 확인 → clone/pull → 데이터셋 5종 prefetch →
학습 잡 제출. 체이닝은 없다. 다섯 셀이 각각 2~16h 라 한 잡에 하나씩 들어간다.

**한 번 던지면 18런(6셀 × 3시드)이 전부 큐에 들어간다.** 기본은 GPU 한 장에 3학습,
동시 2잡 — 즉 **GPU 2장 × 3학습 = 동시 6개**, 세 웨이브로 끝난다.

| cell | task | 조건 | frames | steps | 소요 |
|---|---|---|---|---|---|
| 0 | push_button | A1 | 11,320 | 8,800 | ~0.8h |
| 1 | pick_place | A1 | 28,459 | 22,200 | ~2.0h |
| 2 | sort_by_color | A1 | 74,322 | 58,050 | ~5.1h |
| 3 | push_button | A2 | 11,380 | 8,850 | ~0.8h |
| 4 | pick_place | A2 | 31,526 | 24,600 | ~2.2h |
| 5 | sort_by_color | A2 | 74,921 | 58,500 | ~5.2h |
| 6 | pick_place `_via4cm` | A1 | 28,530 | 22,250 | ~2.7h |
| 7 | pick_place `_via4cm` | A2 | 28,755 | 22,450 | ~2.7h |
| 8 | sort_by_color `_via4cm` | A1 | 74,505 | 58,200 | ~7.0h |
| 9 | sort_by_color `_via4cm` | A2 | 74,827 | 58,450 | ~7.0h |
| 10 | sort_by_color (A0, vanilla CaP) | A0 | 74,450 | 58,150 | ~7.0h |

셀 6~9 는 `_via4cm` 변형이라 **기본에 포함되지 않는다** (기본은 0-5).
태스크별로 전용 진입점이 있다 — 붙여넣고 던지면 끝, 수정할 게 없다:

```
cluster/main_job_via4cm.sbatch          ← 셀 6,7 pick_place    × 3시드 = 6런
cluster/main_job_via4cm_sort.sbatch     ← 셀 8,9 sort_by_color × 3시드 = 6런
cluster/main_job_ablation_sort.sbatch   ← 셀 10  sort_by_color A0 × 3시드 = 3런
```

셀 10 은 `ablation_sort_by_color_100_10fps` (A0 / vanilla CaP, 섭동 없음) 이다.
A0 vs A1 vs A2 비교의 기준선이고, 이름 규칙도 phase1 이 아니라 ablation_study 것을
따른다 — `smolvla_ablation_<task>_<seed>_10fps`. 본체의 `DS_SET` / `NAME_SET` 가
그 예외를 처리한다.

둘은 잡 이름이 달라(`smolvla-via4cm-pp` / `smolvla-via4cm-sort`) 동시에 큐에 넣어도
중복 가드가 서로를 막지 않는다.

`_via4cm` push_button 데이터셋은 없다 (2026-09-01 확인).

셀 1·4 를 대체하는 게 아니라 **비교 대상**이므로 별도 셀로 둔다. 모델 이름은
`smolvla_phase1_pick_place_A1_via4cm_<seed>_10fps` 로 조건 뒤에 변형이 붙는다.

```
wave 1   task0 (cell 0,1,2 · s1000)   task1 (cell 3,4,5 · s1000)
wave 2   task2 (cell 0,1,2 · s2000)   task3 (cell 3,4,5 · s2000)
wave 3   task4 (cell 0,1,2 · s3000)   task5 (cell 3,4,5 · s3000)
```

총 ~48 GPU-hours. 일부 셀만 돌리려면 `PHASE1_CELLS`, 한 GPU 당 개수는 `PACK`,
패킹 자체를 끄려면 `PACKED=0`:

```
sbatch --export=ALL,PHASE1_CELLS=0,1,3 cluster/main_job_phase1.sbatch
```

> ⚠️ **프레임 수는 움직인다.** 여섯 데이터셋 전부 2026-08-27~29 사이 재수집됐고 넷이
> 크기가 바뀌었다 (pick_place A1 31,744 → 28,459 / pick_place A2 30,370 → 31,526).
> 아무 공지도 없었다. 본체가 스테이징된 사본에서 예산을 재계산해 표와 어긋나면 학습 전에
> 죽는다 — 잘못된 epoch 으로 학습된 셀은 형제 셀과 비교 불가고 로그에는 아무 표시도
> 안 남기 때문이다. 이 FATAL 이 뜨면 case 블록의 숫자를 갱신할 것.

### C. SmolVLA — benchmark_table Ours arm (11 태스크 × 3 시드)

| # | 파일 | 비고 |
|---|---|---|
| 1 | **`main_job_bench_ours.sbatch`** | **이것 하나만 던진다** |

**array 태스크 하나 = 데이터셋 하나의 시드 3개(1000/2000/3000).** 한 GPU 에 `PACK` 개씩
웨이브로 돌린다 — 기본 `PACK=2`, 세션마다 `num_workers=8`. 시드 3개면 **2개 → 1개** 두
웨이브이고, 태스크가 끝나면 한 행이 통째로 완성되는 성질은 유지된다. 태스크가 끝나면 benchmark table 의 한 행이
통째로 완성된다 (세 행의 1/3 씩이 아니라).

**GPU 는 절대 2장을 넘지 않는다** — `--array=0-10%2` 가 동시 태스크를 2개로 묶고 태스크당
`--gres=gpu:pro6000:1` 이다.

`--cpus-per-task=18` (8×2 + 메인). `%2` 면 노드 64 중 36.

**PACK=3 을 쓰지 않는 이유.** 8/29 에 3개 패킹이 `updt_s 0.43` (6.98 step/s, 단독 3.14
대비 명백한 이득) 이었는데, 그건 **노드가 반쯤 놀 때** 측정한 값이다 — CPU 부족으로 array
태스크 하나만 돌고 있었다. 두 태스크가 다 뜬 상태(노드에 6런)에서 같은 설정이 **0.94**,
즉 3.19 step/s 로 이득이 전부 사라졌다. 단독으로 하나씩 돌리는 것과 같다.

`PACK=2` + workers 8 이 그 중간이다. 첫 로그 몇 줄의 `updt_s` 가 0.94 보다 낮은지로
바로 판정된다.

```
array 0  stack_2_cubes    3  push_button     6  close_box        9  push_cube
array 1  pick_place       4  extract_cube    7  turn_on_lever   10  pull_cube
array 2  sort_by_color    5  open_box        8  turn_off_lever
```

앞 세 개가 우선순위다. 순서를 바꾸려면 `train_bench_ours.sbatch` 의 `TASKS=(...)` 만
고치면 된다 — 순서가 사는 곳은 거기 하나다.

**step 예산이 파일에 없다.** 데이터셋마다 스테이징한 사본에서 `floor(frames/64) × 50` 을
런타임에 유도하고, 학습 전에 로그에 찍는다. 다른 잡들은 프레임 수를 박아두고 대조하지만
이쪽은 그럴 수가 없다 — 다섯 개가 아직 수집 중이다. 대신 run dir 의 `.dataset_frames`
스탬프가 데이터셋이 바뀐 뒤의 재실행을 계속 막는다.

**없는 데이터셋은 실패가 아니라 건너뛴다.** prefetch 는 있는 것만 받고 없는 것은 알리며,
해당 array 태스크는 안내만 남기고 `exit 0` 한다. 수집이 끝날 때마다 같은 파일을 다시
던지면 된다 — 이미 끝난 런은 체크포인트를 이어받아 1분 만에 재 push 하고 넘어간다.

**11/11 전부 준비됐다** (2026-09-06). 벽시계는 시드 3개를 한 GPU 에서 동시에 돌린
기준이다 — 순차가 아니라 `updt_s 0.43` 로 셋이 같이 돈다.

| array | task | frames | steps | 태스크 벽시계 |
|---|---|---|---|---|
| 0 | stack_2_cubes | 37,889 | 29,600 | 3.5h |
| 1 | pick_place | 29,269 | 22,850 | 2.7h |
| 2 | sort_by_color | 74,952 | 58,550 | **7.0h** |
| 3 | push_button | 11,380 | 8,850 | 1.1h |
| 4 | extract_cube | 31,555 | 24,650 | 2.9h |
| 5 | open_box | 29,116 | 22,700 | 2.7h |
| 6 | close_box | 28,246 | 22,050 | 2.6h |
| 7 | turn_on_lever | 21,927 | 17,100 | 2.0h |
| 8 | turn_off_lever | 21,682 | 16,900 | 2.0h |
| 9 | push_cube | 21,629 | 16,850 | 2.0h |
| 10 | pull_cube | 32,039 | 25,000 | 3.0h |

합계 **31.7 GPU-h**, `%2` 로 GPU 2장이면 **≈16h**. 최장 태스크가 7h 이라 walltime 1일로
충분하다. 위 steps 는 참고용이고, 실제 값은 각 런이 스테이징한 사본에서 다시 유도한다.

모델 레포는 `smolvla_<task>_ours_<seed>_10fps` — CaP arm 과 같은 규칙의 `ours` 쪽이다.

### ~~C. SmolVLA — redundancy, compute-matched~~ (폐기 2026-09-06)

| # | 파일 | 비고 |
|---|---|---|
| 1 | **`main_job_redundancy_cm.sbatch`** | **이것 하나만 던진다** |

**여기만 step 규칙이 다르다.** 다른 실험은 전부 `floor(frames/64)×50` 으로 데이터셋마다
50 epoch 을 맞추지만, 이 실험은 **네 개가 모두 같은 29,100 step** 을 돈다 — per10(가장 큰
셋)의 50 epoch 에 해당하는 값이다.

> **이 컨벤션은 폐기됐다.** 실험이 끝났고 결과 7건은 Hub 에 있다. 새 실험의 본으로 삼지
> 말 것 — 그룹별 고정 step, 단일 시드, 이름의 `<steps>step` 은 전부 이 실험 전용이다.
> 현재 컨벤션은 **50 epoch / batch 64 / 시드 1000·2000·3000** 이고 D 섹션을 따른다.
> 파일은 재현을 위해 남겨두되 헤더에 RETIRED 를 명시했다.

### D. SmolVLA — benchmark_table CaP arm (4 태스크 × 3 시드)

| # | 파일 | 비고 |
|---|---|---|
| 1 | **`main_job_bench_cap.sbatch`** | **이것 하나만 던진다.** 0번(git_sync)도 필요 없다 |

한 번 던지면 12런이 전부 큐에 들어간다. `cell = idx % 4`, `seed = 1000 + idx/4`.

| cell | task | frames | steps | 실측 소요 |
|---|---|---|---|---|
| 0 | pull_cube | 31,714 | 24,750 | ~2.2h |
| 1 | stack_2_cubes | 37,245 | 29,050 | ~2.6h |
| 2 | turn_off_lever | 21,317 | 16,650 | ~1.5h |
| 3 | turn_on_lever | 20,962 | 16,350 | ~1.4h |

```
idx 0-3   seed 1000     총 ~23 GPU-hours
idx 4-7   seed 2000     %2 (GPU 2장) 로 ~12h
idx 8-11  seed 3000
```

모델 레포는 `smolvla_<task>_cap_<seed>_10fps` — `configs/benchmark_table/README.md` 의
명명 규칙 그대로다.

**기본이 GPU 한 장에 2개씩이다.** GPU 2장 × 2학습 = 동시 4개, 12런이 3웨이브로 끝난다.
한 장에 하나씩 돌리려면 `PACKED=0`:

```
sbatch --export=ALL,PACKED=0 cluster/main_job_bench_cap.sbatch
```

단 **빨라진다는 보장은 없다.** A6000 에서는 두 개를 한 GPU 에 올리자 1.5 → 2.9/3.8 s/step
으로 총 처리량이 그대로였다. pro6000 은 다를 수 있으니 `updt_s` 를 0.318 과 비교할 것 —
0.64 미만이면 이득, 초과면 손해다. 그리고 두 학습이 한 잡이라 하나가 죽으면 둘 다 죽는다.

**A 의 `main_job.sbatch` 와 혼동하지 말 것.** 그쪽은 다른 네 개의 `*_cap_10fps` 를
**300 epoch / batch 32** 로 돌리고 이름도 `smolvla_<ds>_300ep` 다. 별개의 일회성 실험이고
benchmark table 셀이 아니다. 이쪽은 **50 epoch / batch 64** — pick_place·push_button·
sort_by_color 의 CaP 셀이 재사용하는 `ablation_*` 모델과 같은 예산이어야 표의 행이
비교 가능하기 때문이다.

### E. SmolVLA — towel_fold01, delta action

| # | 파일 | 비고 |
|---|---|---|
| 1 | `prefetch_delta.sbatch` | `_delta` 데이터셋 4종 |
| 2 | `train_delta_step{0,1,2,3}.sbatch` | 스텝 하나만 띄울 때. 이쪽을 기본으로 |
| 2' | `train_delta.sbatch` | 여러 스텝을 한 번에 (`--array`). 이미 도는 스텝은 넣지 말 것 |

### F. SmolVLA — towel_fold01 / lekiwi, absolute action (베이스라인)

| # | 파일 | 비고 |
|---|---|---|
| 1 | `prefetch.sbatch` | 원본 데이터셋 5종 |
| 2 | `train.sbatch` | `--array` 로 0=step0 … 4=lekiwi |

## 던지기 전 항상 확인

1. `~/.wandb_token`, `~/.hf_token` 이 클러스터 `$HOME` 에 있고 **비어있지 않을 것**.
   OOD Files 앱으로 만든다. 없으면 학습 잡이 시작 직후 FATAL.
2. 수정한 스크립트가 **GitHub `main` 에 push 되어 있을 것**. 잡은 `$HOME/lerobot` 을
   `git pull --ff-only` 로 갱신해서 `_body.sh` 와 `src/lerobot` 을 읽는다.
3. (E·F) prefetch 를 **완주**시킨 뒤에 학습을 던질 것. 순서가 뒤바뀌면
   `FATAL: $HOME/datasets/... not found`. A~D 는 `main_job*.sbatch` 가 순서를 보장한다.

## 자동 pull 이 닿지 않는 곳 — 함정

`*_body.sh` 와 `src/lerobot` 은 매 잡마다 자동으로 최신이 된다. 그러나 **Job Composer 에
붙여넣은 wrapper 자신은 아니다.** SLURM 은 제출 시점에 스크립트를 복사해 두므로, 제출한
잡이 실행하는 wrapper 는 "그때 붙여넣은 그 텍스트"다.

- wrapper(`*.sbatch`)를 고쳤으면 → **다시 붙여넣어 제출**해야 반영된다.
- body(`*_body.sh`)나 `src/lerobot` 을 고쳤으면 → push 만 하면 다음 잡부터 반영된다.

`train_cap300.sbatch` 의 2번 링크는 예외다. `$HOME/lerobot/cluster/` 의 사본을 제출하므로
pull 된 최신 wrapper 를 쓴다.

## 다른 레인 — 클러스터가 아닌 것

| 위치 | 무엇 |
|---|---|
| 레포 루트 `train_smolvla_*.sh`, `train_dp_towel.sh` | conda 로컬 단일 노드용. `#SBATCH` 없음. **클러스터 잡 아님** |
| `local/` | 로컬 dev 박스용 준비/검증 도구 (`preflight.py`, `prepare_delta_dataset.sh` 등) |
