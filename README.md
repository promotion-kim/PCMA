# PCMA DPO 환경 구성 가이드: B200 / RTX PRO 6000 / Blackwell GPU

이 문서는 PCMA의 `experiment-DPO` 코드를 NVIDIA Blackwell 계열 GPU에서 실행하기 위한 환경 구성 절차입니다.

대상 GPU 예시:

- NVIDIA B200 / GB200 계열: 보통 `sm_100`
- NVIDIA RTX PRO 6000 Blackwell / RTX 50-series 계열: 보통 `sm_120`

기존 PCMA DPO 환경은 대체로 `torch==2.1.0+cu118`, `triton==2.1.0`, `flash-attn==2.3.2` 같은 오래된 CUDA 11.8 계열 스택을 가정합니다. 이 조합은 Blackwell GPU에서 다음과 같은 에러를 만들 수 있습니다.

```text
NVIDIA ... Blackwell ... CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

또는:

```text
NCCL error ...
Cuda failure 'named symbol not found'
```

따라서 이 README에서는 다음 원칙을 따릅니다.

1. `torch`, `torchvision`, `torchaudio`는 CUDA 12.8 PyTorch wheel index에서 별도로 설치한다.
2. PCMA 의존성은 별도 파일인 `requirements_blackwell.txt`로 설치한다.
3. `requirements_blackwell.txt` 설치 시 반드시 `--no-deps`를 사용해서 pip가 torch를 바꾸지 못하게 한다.
4. `triton`은 직접 pin하지 않는다. PyTorch가 요구하는 버전을 그대로 사용한다.
5. `sympy==1.12`로 내리지 않는다. 최신 PyTorch는 보통 `sympy>=1.13.3`을 요구한다.
6. `~/.local/bin/accelerate`나 `/usr/bin/python`이 끼어들지 않도록 conda env Python을 강제한다.

---

## 0. 파일 구성

PCMA의 `experiment-DPO` 폴더 안에 다음 파일을 둔다고 가정합니다.

```text
PCMA/
└── experiment-DPO/
    ├── requirements.txt
    ├── requirements_blackwell.txt
    ├── scripts/
    │   └── dpo/
    │       └── run_hh.sh
    └── ...
```

이 README에서는 **원본 `requirements.txt`를 직접 설치하지 않습니다.**

사용할 파일은 새로 제공하는:

```text
requirements_blackwell.txt
```

입니다.

---

## 1. 기존 `dpo` 환경 제거

기존 환경이 꼬였을 가능성이 크므로, 가능하면 새로 만드는 것을 권장합니다.

```bash
conda deactivate
conda env remove -n dpo -y
```

---

## 2. 새 `dpo` 환경 생성

```bash
conda create -n dpo python=3.10 -y
conda activate dpo

python -m pip install -U pip setuptools wheel packaging ninja
```

---

## 3. `~/.local` 패키지 오염 방지

이전 에러에서 `(dpo)`가 켜져 있는데도 `~/.local/bin/accelerate`와 `/usr/bin/python`이 사용된 적이 있었습니다. 이를 막기 위해 conda env 활성화 시 user site-package를 비활성화합니다.

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh" <<'EOF'
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
EOF

conda deactivate
conda activate dpo
hash -r
```

확인:

```bash
which python
python -V
```

정상 예시:

```text
/home/sjkim/anaconda3/envs/dpo/bin/python
Python 3.10.x
```

---

## 4. Blackwell용 PyTorch 설치

CUDA 12.8 PyTorch wheel을 설치합니다.

```bash
python -m pip install --no-cache-dir \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

설치 후 확인:

```bash
python - <<'PY'
import torch

print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("available =", torch.cuda.is_available())
print("device =", torch.cuda.get_device_name(0))
print("capability =", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA matmul OK")
PY
```

정상 예시:

```text
torch = 2.11.0+cu128
cuda = 12.8
available = True
device = NVIDIA ...
capability = (10, 0)   # B200/GB200 계열일 수 있음
# 또는
capability = (12, 0)   # RTX PRO 6000 Blackwell 계열일 수 있음
CUDA matmul OK
```

여기서 실패하면 PCMA 의존성을 설치하지 말고 PyTorch/CUDA 문제부터 해결해야 합니다.

---

## 5. PCMA 폴더로 이동

이미 clone되어 있다면:

```bash
cd /home/sjkim/PCMA/experiment-DPO
```

없다면:

```bash
cd /home/sjkim
git clone https://github.com/promotion-kim/PCMA.git
cd PCMA/experiment-DPO
```

---

## 6. `requirements_blackwell.txt` 배치

`experiment-DPO` 폴더에 다음 파일을 저장합니다.

```text
requirements_blackwell.txt
```

파일 내용은 별도로 제공된 `requirements_blackwell.txt`를 그대로 사용하면 됩니다.

중요하게도 이 파일에는 다음을 포함하지 않습니다.

```text
torch
torchvision
torchaudio
triton
sympy==1.12
```

이유:

- `torch`, `torchvision`, `torchaudio`는 CUDA 12.8 wheel index에서 이미 설치했기 때문입니다.
- `triton`은 PyTorch가 요구하는 버전을 그대로 써야 합니다.
- `sympy==1.12`는 최신 PyTorch의 요구사항과 충돌할 수 있습니다.

---

## 7. PCMA 의존성 설치

반드시 `--no-deps`를 사용합니다.

```bash
python -m pip install --no-deps -r requirements_blackwell.txt
```

`--no-deps`를 빼면 pip가 `accelerate`, `peft`, `trl` 등의 dependency를 다시 해석하면서 `torch==2.11.0+cu128`을 PyPI의 다른 torch 버전으로 바꿀 수 있습니다. 실제로 이런 경우 `torch==2.4.1` 같은 버전으로 내려가면서 Blackwell에서 다시 CUDA/NCCL 에러가 발생할 수 있습니다.

---

## 8. 전체 환경 점검

```bash
python - <<'PY'
import sys
import torch
import numpy
import pandas
import pyarrow
import scipy
import datasets
import fsspec
import transformers
import accelerate
import peft
import trl

print("python =", sys.executable)
print("torch =", torch.__version__, "cuda =", torch.version.cuda)
print("numpy =", numpy.__version__)
print("pandas =", pandas.__version__)
print("pyarrow =", pyarrow.__version__)
print("scipy =", scipy.__version__)
print("datasets =", datasets.__version__)
print("fsspec =", fsspec.__version__)
print("transformers =", transformers.__version__)
print("accelerate =", accelerate.__version__)
print("peft =", peft.__version__)
print("trl =", trl.__version__)

print("cuda available =", torch.cuda.is_available())
print("device =", torch.cuda.get_device_name(0))
print("capability =", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA matmul OK")
PY
```

기대되는 주요 버전:

```text
torch = ...+cu128
cuda = 12.8
numpy = 1.26.4
fsspec = 2023.6.0
datasets = 2.14.5
transformers = 4.34.1
accelerate = 0.23.0
peft = 0.5.0
trl = 0.7.11
CUDA matmul OK
```

추가로 dependency conflict를 확인합니다.

```bash
python -m pip check
```

가능하면 여기서 `torch`, `numpy`, `fsspec`, `datasets`, `pandas`, `pyarrow`, `scipy` 관련 critical conflict가 없어야 합니다.

---

## 9. Hugging Face gated model 접근 설정

`meta-llama/Llama-3.1-8B-Instruct`를 사용하는 경우, Hugging Face gated repo 접근 권한이 필요합니다.

```bash
huggingface-cli login
huggingface-cli whoami
```

또는:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

토크나이저 로딩 테스트:

```bash
python - <<'PY'
from transformers import AutoTokenizer

model = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print("tokenizer loaded:", tok.__class__)
PY
```

여기서 `401 Unauthorized` 또는 `GatedRepoError`가 나면 환경 문제가 아니라 Hugging Face 접근 권한 문제입니다.

---

## 10. `scripts/dpo/run_hh.sh` 수정

bare `accelerate launch`를 쓰면 `~/.local/bin/accelerate`가 잡힐 수 있습니다. `scripts/dpo/run_hh.sh`에서 다음 줄을 찾습니다.

```bash
LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=${NUM_PROCESSES} --main_process_port ${PORT}"
```

아래처럼 바꿉니다.

```bash
export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${PATH}"

PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
LAUNCH="${PYTHON_BIN} -m accelerate.commands.launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=${NUM_PROCESSES} --main_process_port ${PORT}"
```

이렇게 하면 현재 conda env의 Python으로 accelerate가 실행됩니다.

확인:

```bash
which python
which accelerate
head -n 1 "$(which accelerate)"
```

정상 예시:

```text
/home/sjkim/anaconda3/envs/dpo/bin/python
/home/sjkim/anaconda3/envs/dpo/bin/accelerate
```

비정상 예시:

```text
/home/sjkim/.local/bin/accelerate
#!/usr/bin/python
```

---

## 11. DPO-HH 실행

```bash
conda activate dpo
cd /home/sjkim/PCMA/experiment-DPO

export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash scripts/dpo/run_hh.sh helpful
```

다른 objective:

```bash
NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash scripts/dpo/run_hh.sh harmless
NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash scripts/dpo/run_hh.sh humor
```

---

## 12. 단일 GPU 디버깅용 우회

`NUM_PROCESSES=1`인데도 `DistributedDataParallel`/NCCL 쪽에서 문제가 나면, 디버깅 목적으로만 `accelerate launch`를 우회할 수 있습니다.

`run_hh.sh`에서 임시로:

```bash
LAUNCH="python"
```

로 바꿉니다.

이것은 근본 해결책은 아니며, 단일 GPU에서 코드가 어디까지 가는지 확인하기 위한 우회입니다.

---

## 13. flash-attn은 처음에는 설치하지 않기

기존 README 또는 오래된 환경에서는 다음을 안내할 수 있습니다.

```bash
pip install flash-attn==2.3.2 --no-build-isolation
```

Blackwell 환경에서는 우선 설치하지 마세요.

현재 `dpo_hh.py`는 기본적으로:

```python
use_flash_attention_2 = False
```

이므로 flash-attn 없이 먼저 DPO 학습이 되는지 확인하는 것이 안전합니다.

---

## 14. 자주 발생한 에러와 해결

### 14.1 `torch.library has no attribute impl_abstract`

가능한 원인:

- conda env가 아니라 `~/.local`의 `bitsandbytes`/`accelerate`/Python이 사용됨
- `~/.local/bin/accelerate`가 실행됨

해결:

```bash
export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

which python
which accelerate
```

---

### 14.2 `GatedRepoError: 401 Unauthorized`

가능한 원인:

- Hugging Face login이 안 됨
- 해당 계정이 Llama gated repo 접근 승인을 받지 않음
- token에 read 권한이 없음

해결:

```bash
huggingface-cli login
huggingface-cli whoami
```

그리고 tokenizer loading test를 실행합니다.

---

### 14.3 `NCCL error ... Cuda failure 'named symbol not found'`

가능한 원인:

- Blackwell을 지원하지 않는 PyTorch가 설치됨
- 예: `torch==2.1.0+cu118`
- 또는 pip resolver가 `torch==2.11.0+cu128`을 다른 버전으로 바꿈

해결:

```bash
python -m pip install --force-reinstall --no-cache-dir \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

그 다음 다시:

```bash
python -m pip install --no-deps -r requirements_blackwell.txt
```

---

### 14.4 `datasets requires fsspec<2023.9.0`

가능한 원인:

- torch 재설치 과정에서 `fsspec`이 최신 버전으로 올라감

해결:

```bash
python -m pip install --no-deps --force-reinstall fsspec==2023.6.0
```

또는 전체 requirements 재설치:

```bash
python -m pip install --no-deps -r requirements_blackwell.txt
```

---

### 14.5 `pandas/pyarrow/scipy requires numpy<2`

가능한 원인:

- torch 또는 torchvision 재설치 과정에서 `numpy`가 2.x로 올라감

해결:

```bash
python -m pip install --no-deps --force-reinstall numpy==1.26.4
```

또는 전체 requirements 재설치:

```bash
python -m pip install --no-deps -r requirements_blackwell.txt
```

---

## 15. 절대 하지 말아야 할 것

Blackwell용 torch를 설치한 뒤 아래 명령을 실행하지 마세요.

```bash
python -m pip install -r requirements.txt
```

또한 아래처럼 `--no-deps` 없이 새 requirements를 설치하지 마세요.

```bash
python -m pip install -r requirements_blackwell.txt
```

반드시:

```bash
python -m pip install --no-deps -r requirements_blackwell.txt
```

를 사용하세요.

---

## 16. 최소 one-shot setup

아래는 깨끗한 환경에서 한 번에 구성하는 예시입니다.

```bash
conda deactivate
conda env remove -n dpo -y

conda create -n dpo python=3.10 -y
conda activate dpo

python -m pip install -U pip setuptools wheel packaging ninja

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh" <<'EOF'
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
EOF

conda deactivate
conda activate dpo
hash -r

python -m pip install --no-cache-dir \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

cd /home/sjkim/PCMA/experiment-DPO

python -m pip install --no-deps -r requirements_blackwell.txt

python - <<'PY'
import sys
import torch
import numpy
import datasets
import fsspec
import transformers
import accelerate
import peft
import trl

print("python =", sys.executable)
print("torch =", torch.__version__, "cuda =", torch.version.cuda)
print("numpy =", numpy.__version__)
print("datasets =", datasets.__version__)
print("fsspec =", fsspec.__version__)
print("transformers =", transformers.__version__)
print("accelerate =", accelerate.__version__)
print("peft =", peft.__version__)
print("trl =", trl.__version__)

print("cuda available =", torch.cuda.is_available())
print("device =", torch.cuda.get_device_name(0))
print("capability =", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA matmul OK")
PY
```

---

## 17. 최종 확인 규칙

실행 전 항상 이 두 가지를 확인합니다.

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
```

출력은 반드시 CUDA 12.8 빌드여야 합니다.

```text
...+cu128 12.8
```

그리고:

```bash
which python
which accelerate
```

둘 다 conda env 내부를 가리켜야 합니다.

```text
/home/sjkim/anaconda3/envs/dpo/bin/python
/home/sjkim/anaconda3/envs/dpo/bin/accelerate
```
