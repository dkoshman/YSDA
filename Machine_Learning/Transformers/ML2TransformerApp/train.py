from constants import TRAINER_DIR, TOKENIZER_PATH, DATAMODULE_PATH, WANDB_DIR, RESOURCES
from data_generator import generate_data
from data_preprocessing import LatexImageDataModule
from model import Transformer
from utils import LogImageTexCallback, average_checkpoints

import argparse
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch


# TODO: make label smoothing scale with random magnitude, generate photorealistic images with notebook background,
# write own torch multiprocessing


def check_setup():
    # Disabling tokenizers parallelism because it can't be used before forking and I didn't bother to figure it out
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not os.path.isfile(DATAMODULE_PATH):
        print("Generating default datamodule")
        datamodule = LatexImageDataModule(
            image_width=1024, image_height=128, batch_size=16, random_magnitude=5
        )
        torch.save(datamodule, DATAMODULE_PATH)
    if not os.path.isfile(TOKENIZER_PATH):
        print("Generating default tokenizer")
        datamodule = torch.load(DATAMODULE_PATH)
        datamodule.train_tokenizer()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflow: generate dataset, create datamodule, train model",
        allow_abbrev=True,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "gpus",
        type=int,
        help=f"Ids of gpus in range 0..{torch.cuda.device_count() - 1} to train on, "
        "if not provided,\nthen trains on cpu. To see current gpu load, run nvtop",
        nargs="*",
    )
    parser.add_argument(
        "-l",
        "-log",
        help="Whether to save logs of run to w&b logger, default False",
        default=False,
        action="store_true",
        dest="log",
    )
    parser.add_argument(
        "-m",
        "-max-epochs",
        help="Limit the number of training epochs",
        type=int,
        dest="max_epochs",
    )

    data_args = ["size", "depth", "length", "fraction"]
    parser.add_argument(
        "-n",
        metavar=tuple(map(str.upper, data_args)),
        nargs=4,
        dest="data_args",
        type=lambda x: int(x) if x.isdigit() else float(x),
        help="Clear old dataset, create new and exit, args:"
        "\nsize\tsize of new dataset"
        "\ndepth\tmax_depth scope depth of generated equation, no less than 1"
        "\nlength\tlength of equation will be in range length/2..length"
        "\nfraction\tfraction of tex vocab to sample tokens from, float in range 0..1",
    )

    datamodule = torch.load(DATAMODULE_PATH)
    datamodule_args = ["image_width", "image_height", "batch_size", "random_magnitude"]
    parser.add_argument(
        "-d",
        metavar=tuple(map(str.upper, datamodule_args)),
        nargs=4,
        dest="datamodule_args",
        type=int,
        help="Create new datamodule and exit, current parameters:\n"
        + "\n".join(f"{arg}\t{datamodule.hparams[arg]}" for arg in datamodule_args),
    )

    transformer_args = [
        ("num_encoder_layers", 6),
        ("num_decoder_layers", 6),
        ("d_model", 512),
        ("nhead", 8),
        ("dim_feedforward", 2048),
        ("dropout", 0.1),
    ]
    parser.add_argument(
        "-t",
        metavar=tuple(args[0].upper() for args in transformer_args),
        dest="transformer_args",
        nargs=len(transformer_args),
        help="Transformer init args, default values:\n"
        + "\n".join(f"{k}\t{v}" for k, v in transformer_args),
    )

    args = parser.parse_args()
    if args.data_args:
        args.data_args = dict(zip(data_args, args.data_args))
    if args.datamodule_args:
        args.datamodule_args = dict(zip(datamodule_args, args.datamodule_args))

    if args.transformer_args:
        args.transformer_args = dict(
            zip(list(zip(*transformer_args))[0], args.transformer_args)
        )
    else:
        args.transformer_args = dict(transformer_args)

    return args


def main():
    check_setup()
    args = parse_args()
    if args.data_args:
        generate_data(
            examples_count=args.data_args["size"],
            max_depth=args.data_args["depth"],
            equation_length=args.data_args["length"],
            distribution_fraction=args.data_args["fraction"],
        )
        return

    if args.datamodule_args:
        datamodule = LatexImageDataModule(
            image_width=args.datamodule_args["image_width"],
            image_height=args.datamodule_args["image_height"],
            batch_size=args.datamodule_args["batch_size"],
            random_magnitude=args.datamodule_args["random_magnitude"],
        )
        datamodule.train_tokenizer()
        tex_tokenizer = torch.load(TOKENIZER_PATH)
        print(f"Vocabulary size {tex_tokenizer.get_vocab_size()}")
        torch.save(datamodule, DATAMODULE_PATH)
        return

    datamodule = torch.load(DATAMODULE_PATH)
    tex_tokenizer = torch.load(TOKENIZER_PATH)
    logger = None
    callbacks = []
    if args.log:
        logger = WandbLogger(f"img2tex", save_dir=WANDB_DIR, log_model=True)
        callbacks = [
            LogImageTexCallback(logger, top_k=10, max_length=100),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                save_top_k=10,
                every_n_train_steps=5,
                monitor="val_loss",
                mode="min",
                filename="img2tex-{epoch:02d}-{val_loss:.2f}",
            ),
        ]

    trainer = Trainer(
        default_root_dir=TRAINER_DIR,
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus else "cpu",
        gpus=args.gpus,
        logger=logger,
        strategy="ddp_find_unused_parameters_false",
        enable_progress_bar=True,
        callbacks=callbacks,
    )

    transformer = Transformer(
        num_encoder_layers=args.transformer_args["num_encoder_layers"],
        num_decoder_layers=args.transformer_args["num_decoder_layers"],
        d_model=args.transformer_args["d_model"],
        nhead=args.transformer_args["nhead"],
        dim_feedforward=args.transformer_args["dim_feedforward"],
        dropout=args.transformer_args["dropout"],
        image_width=datamodule.hparams["image_width"],
        image_height=datamodule.hparams["image_height"],
        tgt_vocab_size=tex_tokenizer.get_vocab_size(),
        pad_idx=tex_tokenizer.token_to_id("[PAD]"),
    )

    trainer.fit(transformer, datamodule=datamodule)
    trainer.test(transformer, datamodule=datamodule)

    if args.log and len(os.listdir(trainer.checkpoint_callback.dirpath)):
        transformer = average_checkpoints(
            model_type=Transformer, checkpoints_dir=trainer.checkpoint_callback.dirpath
        )
        transformer_path = os.path.join(RESOURCES, f"model_{trainer.logger.version}.pt")
        transformer.eval()
        transformer.freeze()
        torch.save(transformer.state_dict(), transformer_path)
        print(f"Transformer ensemble saved to '{transformer_path}'")


if __name__ == "__main__":
    main()
