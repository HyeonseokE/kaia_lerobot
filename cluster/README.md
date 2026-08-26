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

C 와 D 는 **0번을 먼저 한 번** 돌려야 한다. `$HOME/lerobot` 체크아웃이 없으면 학습 잡이
`FATAL: ... _body.sh not found` 로 즉사한다. A·B 는 `main_job*.sbatch` 가 클론까지 하므로
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

**한 번 던지면 15런(5셀 × 3시드)이 전부 큐에 들어간다.** array 인덱스가 셀과 시드를
같이 인코딩한다 — `cell = idx % 5`, `seed = 1000 + idx/5`.

| cell | task | 조건 | frames | steps | 실측 소요 |
|---|---|---|---|---|---|
| 0 | push_button | A1 | 11,299 | 8,800 | ~47분 |
| 1 | pick_place | A1 | 31,744 | 24,800 | ~2.2h |
| 2 | sort_by_color | A1 | 74,255 | 58,000 | ~5.1h |
| 3 | push_button | A2 | 11,359 | 8,850 | ~47분 |
| 4 | pick_place | A2 | 30,370 | 23,700 | ~2.1h |

```
idx  0-4   seed 1000        전체 ~33 GPU-hours
idx  5-9   seed 2000        GPU 3~4장이면 ~11h
idx 10-14  seed 3000
```

소요 시간은 실측이다 — pro6000 에서 `updt_s:0.318` (batch 64, 카메라 2개, 2026-08-27).
`--time` 은 1일이면 충분하다.

부분 실행은 array 를 직접 지정한다:

```
sbatch --array=5-9 cluster/train_phase1.sbatch       # seed 2000 전부
sbatch --array=2,7,12 cluster/train_phase1.sbatch    # sort_by_color A1, 세 시드
```

**sort_by_color A2 는 아직 수집 전이다.** 위 5셀이 현재 전부다. 수집되면
`train_phase1_body.sh` 의 case 에 행 추가 + 인덱스 산술의 `5` 를 `6` 으로, array 를
`0-17` 로 바꾸면 된다.

**동시 실행 수는 `%N` 으로 정한다.** 잡 하나가 GPU 한 장이므로 `%2` = GPU 2장이다.
현재 `--array=0-14%2`. 파티션 전체를 쓰려면 `%4`.

노드를 지정하려면 제출할 때 `--nodelist` / `--exclude` 를 붙인다:

```
sbatch --nodelist=p1 cluster/train_phase1.sbatch
```

**특정 GPU 인덱스는 지정할 수 없다.** SLURM 이 노드에서 GPU 를 골라 할당하고
`CUDA_VISIBLE_DEVICES` 를 대신 설정한다. 노드가 가장 세밀한 단위다.

`--cpus-per-task=12` 이다 (towel 잡은 16). phase1 은 `num_workers=8` / 카메라 2개라 16은
슬랙이었고, 실제로 슬롯을 하나 까먹었다 — 2026-08-27 pro6000 한 장과 CPU 정확히 16개가
비어 있는데도 task 3 이 `Reason: Resources` 로 못 들어갔다. **`num_workers` 를 올리면
이 값도 같이 올려야 한다.**

**학습 인자는 SCRAPE phase1 컨벤션을 따른다** — batch **64**(32 아님), 50 epoch,
torchcodec, 최종 체크포인트만. Phase-1 은 같은 태스크의 A0/A1/A2 를 비교하는 실험이고,
모든 셀을 동일하게 학습해야 비교가 성립한다. `train_cap300_body.sh` 의 숫자와 "통일"하지
말 것 — step 공식도 다르다 (phase1 은 `floor(frames/64) × 50`, cap300 은 총합 올림).

모델 레포는 `smolvla_phase1_<task>_<조건>_<seed>_10fps`.

### C. SmolVLA — towel_fold01, delta action

| # | 파일 | 비고 |
|---|---|---|
| 1 | `prefetch_delta.sbatch` | `_delta` 데이터셋 4종 |
| 2 | `train_delta_step{0,1,2,3}.sbatch` | 스텝 하나만 띄울 때. 이쪽을 기본으로 |
| 2' | `train_delta.sbatch` | 여러 스텝을 한 번에 (`--array`). 이미 도는 스텝은 넣지 말 것 |

### D. SmolVLA — towel_fold01 / lekiwi, absolute action (베이스라인)

| # | 파일 | 비고 |
|---|---|---|
| 1 | `prefetch.sbatch` | 원본 데이터셋 5종 |
| 2 | `train.sbatch` | `--array` 로 0=step0 … 4=lekiwi |

## 던지기 전 항상 확인

1. `~/.wandb_token`, `~/.hf_token` 이 클러스터 `$HOME` 에 있고 **비어있지 않을 것**.
   OOD Files 앱으로 만든다. 없으면 학습 잡이 시작 직후 FATAL.
2. 수정한 스크립트가 **GitHub `main` 에 push 되어 있을 것**. 잡은 `$HOME/lerobot` 을
   `git pull --ff-only` 로 갱신해서 `_body.sh` 와 `src/lerobot` 을 읽는다.
3. (C·D) prefetch 를 **완주**시킨 뒤에 학습을 던질 것. 순서가 뒤바뀌면
   `FATAL: $HOME/datasets/... not found`. A·B 는 `main_job*.sbatch` 가 순서를 보장한다.

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
