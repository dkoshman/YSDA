from constants import (
    DATA_DIR,
    TOKENIZER_PATH,
    NUM_DATALOADER_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
)

import einops
import os
import pytorch_lightning as pl
import tokenizers
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import tqdm
import re


class TexImageDataset(Dataset):
    """Image and tex dataset."""

    def __init__(self, root_dir, image_transform=None, tex_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and tex files.
            image_transform: callable image preprocessing
            tex_transform: callable tex preprocessing
        """

        torch.multiprocessing.set_sharing_strategy("file_system")
        self.root_dir = root_dir
        self.filenames = sorted(
            set(
                os.path.splitext(filename)[0]
                for filename in os.listdir(root_dir)
                if filename.endswith(".png")
            )
        )
        self.image_transform = image_transform
        self.tex_transform = tex_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.root_dir, filename + ".png")
        tex_path = os.path.join(self.root_dir, filename + ".tex")

        with open(tex_path) as file:
            tex = file.read()
        if self.tex_transform:
            tex = self.tex_transform(tex)

        image = torchvision.io.read_image(image_path)
        if self.image_transform:
            image = self.image_transform(image)

        return {"image": image, "tex": tex}


class BatchCollator(object):
    """Image, tex batch collator"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = [i["image"] for i in batch]
        images = einops.rearrange(images, "b c h w -> b c h w")

        texs = [item["tex"] for item in batch]
        texs = self.tokenizer.encode_batch(texs)
        tex_ids = torch.Tensor([encoding.ids for encoding in texs])
        attention_masks = torch.Tensor([encoding.attention_mask for encoding in texs])

        return {
            "images": images,
            "tex_ids": tex_ids,
            "tex_attention_masks": attention_masks,
        }


class RandomizeImageTransform(object):
    """Standardize image and randomly augment"""

    def __init__(self, width, height, random_magnitude):
        self.transform = T.Compose(
            (
                (lambda x: x)
                if random_magnitude == 0
                else T.ColorJitter(
                    brightness=random_magnitude / 10,
                    contrast=random_magnitude / 10,
                    saturation=random_magnitude / 10,
                    hue=min(0.5, random_magnitude / 10),
                ),
                T.Resize(height, max_size=width),
                T.Grayscale(),
                T.functional.invert,
                T.CenterCrop((height, width)),
                torch.Tensor.contiguous,
                (lambda x: x)
                if random_magnitude == 0
                else T.RandAugment(magnitude=random_magnitude),
                T.ConvertImageDtype(torch.float32),
            )
        )

    def __call__(self, image):
        image = self.transform(image)
        return image


class ExtractEquationFromTexTransform(object):
    """Extracts ...\[ equation \]... from tex file"""

    def __init__(self):
        self.equation_pattern = re.compile(r"\\\[(?P<equation>.*)\\\]", flags=re.DOTALL)
        self.spaces = re.compile(r" +")

    def __call__(self, tex):
        equation = self.equation_pattern.search(tex)
        equation = equation.group("equation")
        equation = equation.strip()
        equation = self.spaces.sub(" ", equation)
        return equation


def generate_tex_tokenizer(dataloader):
    """Returns a tokenizer trained on texs from given dataset"""

    texs = list(
        tqdm.tqdm(
            (batch["tex"] for batch in dataloader),
            "Training tokenizer",
            total=len(dataloader),
        )
    )

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    tokenizer_trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(texs, trainer=tokenizer_trainer)
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")

    return tokenizer


class LatexImageDataModule(pl.LightningDataModule):
    def __init__(self, image_width, image_height, batch_size, random_magnitude):
        super().__init__()

        dataset = TexImageDataset(
            root_dir=DATA_DIR,
            image_transform=RandomizeImageTransform(
                image_width, image_height, random_magnitude
            ),
            tex_transform=ExtractEquationFromTexTransform(),
        )
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            dataset, [len(dataset) * 18 // 20, len(dataset) // 20, len(dataset) // 20]
        )
        self.batch_size = batch_size
        self.save_hyperparameters()

    def train_tokenizer(self):
        tokenizer = generate_tex_tokenizer(
            DataLoader(self.train_dataset, batch_size=32, num_workers=16)
        )
        torch.save(tokenizer, TOKENIZER_PATH)
        return tokenizer

    def _shared_dataloader(self, dataset, **kwargs):
        tex_tokenizer = torch.load(TOKENIZER_PATH)
        collate_fn = BatchCollator(tex_tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_DATALOADER_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
            **kwargs
        )

    def train_dataloader(self):
        return self._shared_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._shared_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._shared_dataloader(self.test_dataset)
