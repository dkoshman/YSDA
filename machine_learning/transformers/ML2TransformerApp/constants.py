PDFLATEX = "/external2/dkkoshman/venv/texlive/2022/bin/x86_64-linux/pdflatex"
GHOSTSCRIPT = "/external2/dkkoshman/venv/local/gs/bin/gs"

DATA_DIR = "local/data"
WANDB_DIR = "local/wandb"
TRAINER_DIR = "local/trainer"

RESOURCES = "resources"
LATEX_PATH = RESOURCES + "/latex.json"
TOKENIZER_PATH = RESOURCES + "/tokenizer.pt"
DATAMODULE_PATH = RESOURCES + "/datamodule.pt"

NUM_DATALOADER_WORKERS = 4
PERSISTENT_WORKERS = True  # whether to shut down workers at the end of epoch
PIN_MEMORY = False  # probably causes cuda oom error if True
