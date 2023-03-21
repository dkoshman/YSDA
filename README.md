# YSDA
Yandex School of Data Analysis materials 

Poetry requirements for torch with cuda on beleriand server:
torch = [
{ url = "https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp310-cp310-linux_x86_64.whl", markers = "sys_platform == 'linux'"},
    { version = "^1.12.0", markers = "sys_platform == 'darwin'" }
]

Beleriand bash script

export JUPYTER_CONFIG_DIR=$VIRTUAL_ENV/etc/jupyter
export JUPYTER_DATA_DIR=$VIRTUAL_ENV/etc/jupyter
export JUPYTER_RUNTIME_DIR=$VIRTUAL_ENV/etc/jupyter
export JUPYTER_PATH=$VIRTUAL_ENV/etc/jupyter

export TRANSFORMERS_CACHE=$VIRTUAL_ENV/venv/.cache

export WANDB_CONFIG_DIR=$VIRTUAL_ENV/.config/wandb
export WANDB_CACHE_DIR=$VIRTUAL_ENV/.cache/wandb
export WANDB_API_KEY=64992677760f66c18bacc1c2fd2661df72c57131
export HF_DATASETS_CACHE=$VIRTUAL_ENV/.cache/huggingface/datasets

export MPLCONFIGDIR=$VIRTUAL_ENV/.config/matplotlib

export TUNE_RESULT_DIR=$VIRTUAL_ENV/ray_results

PATH=/external2/dkkoshman/venv/texlive/2022/bin/x86_64-linux:$PATH
PATH=/external2/dkkoshman/venv/local/gs/bin:$PATH
PATH=/external2/dkkoshman/venv/local/bin:$PATH

