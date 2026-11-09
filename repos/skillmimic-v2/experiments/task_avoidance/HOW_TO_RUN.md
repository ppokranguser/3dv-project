# 처음 코드를 받은 팀원을 위한 실행 방법

이 문서는 기존 `place_book fix` checkpoint에서 Task Avoidance C0/C1 분기 학습을 시작하는 절차다. Linux GPU 머신에서 실행한다.

## 1. 코드 받기

브랜치가 원저장소에 올라간 경우:

```bash
git clone https://github.com/chaseungjoon/3dv-project.git
cd 3dv-project
git checkout experiments/task-avoidance-v2
cd repos/skillmimic-v2
```

이미 코드를 전달받았다면 바로 프로젝트 폴더로 이동한다.

```bash
cd /absolute/path/to/3dv-project/repos/skillmimic-v2
git branch --show-current
git rev-parse HEAD
git status --short
```

예상 branch는 `experiments/task-avoidance-v2`, 최초 구현 commit은 `a8f79344151cbd85de2e34ef16b0bad3360774c6`이다. 팀이 후속 commit을 추가했다면 공유받은 commit을 기록한다.

## 2. GPU 머신에 필요한 것

Git 코드만으로 학습할 수 없다. 기존 GitHub 학습에 사용한 머신에서 다음 항목을 준비한다.

- Isaac Gym과 PyTorch/CUDA가 동작하는 Python
- NVIDIA GPU와 `nvidia-smi`
- ParaHome `place_book` motion 및 transformation data
- robot/book/desk assets
- history encoder checkpoint
- 기존 `place_book fix`, `num_envs=2048` 학습 run

필수 파일의 기본 위치:

```text
experiments/checkpoints/hist_encoder/parahome_hist60.ckpt
skillmimic/data/motions/ParaHome/place_book/002_s6_place_book_2460_2610.pt
skillmimic/data/motions/ParaHome/s10/object_transformations.pkl
skillmimic/data/assets/obj/book_fix.urdf
skillmimic/data/assets/mjcf/mocap_parahome_boxhand_histphase.xml
```

분기 source는 `.pth`만 복사하면 안 된다. 다음 구조와 `run_meta.json`을 함께 보존한다.

```text
place_book_fix_n2048_s0/
├── run_meta.json
└── nn/
    └── SOME_NAME_00001000.pth
```

## 3. Python 환경과 사전 검사

`PY`에는 일반 Python이 아니라 기존 Isaac Gym 환경의 실행 파일 절대 경로를 넣는다.

```bash
export PY=/absolute/path/to/isaacgym/environment/bin/python
export ORCHESTRATOR_PY=python3

"$PY" -c "import isaacgym; import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
bash experiments/scripts/progress_reward/preflight.sh
```

`preflight.sh`가 실패하면 학습을 실행하지 말고 출력된 누락 파일이나 환경 문제를 해결한다. 이 스크립트는 패키지를 설치하지 않는다.

## 4. source checkpoint 찾기

1000 epoch checkpoint 후보를 찾는다.

```bash
find experiments/runs -type f -path '*/nn/*_00001000.pth'
```

기존 run이 다른 디스크에 있다면 해당 경로를 검색한다.

```bash
find /path/to/existing/runs -type f -path '*/nn/*_00001000.pth'
```

`place_book fix`, `num_envs=2048` run을 골라 절대 경로로 지정한다.

```bash
export SOURCE=/absolute/path/to/place_book_fix_n2048_s0/nn/SOME_NAME_00001000.pth
test -f "$SOURCE"
test -f "$(dirname "$(dirname "$SOURCE")")/run_meta.json"
```

`--source`는 다음 조건을 만족해야 한다.

- 파일명이 `_00001000.pth`처럼 8자리 epoch 번호로 끝난다.
- `run_meta.json`이 `clip=place_book`, `method=fix`, `num_envs=2048`을 기록한다.
- model, optimizer, normalizer, ATS, RNG, epoch/frame을 포함한 full checkpoint다.

## 5. 학습 전 GPU 검사

이 단계는 학습하지 않고 4개 환경에서 reward/buffer/부분 reset 경로를 검사한다.

```bash
bash experiments/scripts/task_avoidance/verify_gpu.sh \
  --checkpoint "$SOURCE" --label initial_check --dry-run

bash experiments/scripts/task_avoidance/verify_gpu.sh \
  --checkpoint "$SOURCE" --label initial_check
```

다음 파일의 `status`가 `PASS`인지 확인한다.

```text
experiments/runs/task_avoidance/verify_initial_check/checks/result.json
```

같은 label은 덮어쓰지 않는다. 실패 후 다시 실행할 때는 `initial_check_retry1`처럼 새 label을 사용한다.

## 6. 20 epoch 예비 학습

`--label`은 출력 폴더를 구분하는 영문·숫자·`_`·`-` 이름이다. 먼저 실제 실행 명령만 출력한다.

```bash
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" \
  --label avoidance_pilot \
  --stage smoke \
  --extra-epochs 20 \
  --seeds 0 \
  --dry-run
```

경로와 명령을 확인한 뒤 학습한다.

```bash
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" \
  --label avoidance_pilot \
  --stage smoke \
  --extra-epochs 20 \
  --seeds 0 \
  --execute
```

하나의 source에서 다음 두 run을 순차 학습한다.

- `baseline`: 원 fix reward, progress weight 0
- `progress`: 원 fix reward + progress reward, progress weight 1

source가 1000 epoch이면 목표 checkpoint는 각각 1020 epoch다. 조건별 추가 sample 수는 `20 × 2048 × 32 = 1,310,720`이다.

## 7. 중단된 학습 재개

source, label, epoch, seed를 그대로 유지하고 `--resume`을 추가한다.

```bash
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" \
  --label avoidance_pilot \
  --stage smoke \
  --extra-epochs 20 \
  --seeds 0 \
  --execute \
  --resume
```

조건을 바꿨다면 resume하지 말고 새 label로 시작한다. 목표 checkpoint가 이미 있으면 추가 학습하지 않고 checkpoint를 감사한다.

## 8. 예비 학습 완료 확인

결과 폴더:

```text
experiments/runs/task_avoidance/branch_avoidance_pilot_smoke_baseline_s0_from1000_plus20/
experiments/runs/task_avoidance/branch_avoidance_pilot_smoke_progress_s0_from1000_plus20/
```

두 폴더에서 다음을 확인한다.

```text
experiment.json       status=COMPLETED
completion.json       status=PASS, final_epoch=1020
nn/*_00001020.pth     최종 checkpoint
train.log             학습 로그
checkpoint_audit.log  checkpoint 감사 로그
```

하나라도 FAIL 또는 INCONCLUSIVE이면 본 학습으로 넘어가지 않는다.

## 9. 500 epoch 본 학습

GPU 검사와 예비 학습·평가를 확인하고 실험 비용을 승인한 뒤 실행한다.

```bash
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" \
  --label avoidance_main \
  --stage main \
  --extra-epochs 500 \
  --seeds 0,1,2 \
  --execute
```

`baseline/progress × continuation seeds 0,1,2`, 총 6개 run이 순차 실행된다. source가 1000 epoch이면 목표는 1500 epoch다. 이 seed들은 하나의 source에서 갈라지는 continuation seed이며 독립 from-scratch 학습 seed는 아니다.

중단 시 같은 옵션에 `--resume`을 추가한다.

```bash
bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" \
  --label avoidance_main \
  --stage main \
  --extra-epochs 500 \
  --seeds 0,1,2 \
  --execute \
  --resume
```

## 10. 자주 발생하는 오류

`--source and --label are required`
: 명령에 `--source "$SOURCE" --label avoidance_pilot`을 포함한다.

`Source checkpoint missing`
: `test -f "$SOURCE"`로 GPU 머신의 실제 절대 경로인지 확인한다.

`run_meta.json` 오류
: checkpoint 파일만 복사하지 말고 기존 run 폴더 구조 전체를 가져온다.

`Run exists; explicit matching --resume required`
: 같은 실험을 계속하면 `--resume`, 다른 실험이면 새 label을 사용한다.

`fingerprint changed`
: source/protocol/code/seed/epoch가 기존 run과 다르다. 기존 결과를 덮어쓰지 말고 새 label로 시작한다.

`CUDA unavailable`, `nvidia-smi missing`
: 기존 학습이 동작했던 Linux GPU 환경과 `$PY`를 사용한다.

## 최소 실행 명령

```bash
cd /absolute/path/to/3dv-project/repos/skillmimic-v2
export PY=/absolute/path/to/isaacgym/environment/bin/python
export SOURCE=/absolute/path/to/place_book_fix_n2048_s0/nn/SOME_NAME_00001000.pth

bash experiments/scripts/progress_reward/preflight.sh

bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label avoidance_pilot \
  --stage smoke --extra-epochs 20 --seeds 0 --dry-run

bash experiments/scripts/task_avoidance/run_branches.sh \
  --source "$SOURCE" --label avoidance_pilot \
  --stage smoke --extra-epochs 20 --seeds 0 --execute
```

추가 평가와 donor-prefix 능력 검사는 [README.md](README.md)를 따른다.
