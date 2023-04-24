import collections
import dataclasses
import types

import pytorch_lightning as pl
import torch.utils.data
import transformers

from data import (
    generate_annotated_images,
    get_annotation_ground_truth_str,
    DataItem,
    get_extra_tokens,
    Batch,
    Split,
    BatchCollateFunction,
)
from utils import load_pickle_or_build_object_and_save


@dataclasses.dataclass
class Model:
    processor: transformers.models.donut.processing_donut.DonutProcessor
    tokenizer: transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast
    encoder_decoder: transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder.VisionEncoderDecoderModel
    batch_collate_function: BatchCollateFunction
    config: types.SimpleNamespace


def add_unknown_tokens_to_tokenizer(
        tokenizer, encoder_decoder, unknown_tokens: list[str]
):
    assert set(unknown_tokens) == set(unknown_tokens) - set(
        tokenizer.vocab.keys()
    ), "Tokens are not unknown."

    tokenizer.add_tokens(unknown_tokens)
    encoder_decoder.decoder.resize_token_embeddings(len(tokenizer))


def find_unknown_tokens_for_tokenizer(tokenizer) -> collections.Counter:
    unknown_tokens_counter = collections.Counter()

    for annotated_image in generate_annotated_images():
        ground_truth = get_annotation_ground_truth_str(annotated_image.annotation)

        input_ids = tokenizer(ground_truth).input_ids
        tokens = tokenizer.tokenize(ground_truth, add_special_tokens=True)

        for token_id, token in zip(input_ids, tokens, strict=True):
            if token_id == tokenizer.unk_token_id:
                unknown_tokens_counter.update([token])

    return unknown_tokens_counter


def replace_pad_token_id_with_negative_hundred_for_hf_transformers_automatic_batch_transformation(
        tokenizer, token_ids
):
    token_ids[token_ids == tokenizer.pad_token_id] = -100
    return token_ids


@dataclasses.dataclass
class BatchCollateFunction:
    processor: transformers.models.donut.processing_donut.DonutProcessor
    tokenizer: transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast
    decoder_sequence_max_length: int

    def __call__(self, batch: list[DataItem], split: Split) -> Batch:
        images = [di.image for di in batch]
        images = self.processor(
            images, random_padding=split == Split.train, return_tensors="pt"
        ).pixel_values

        target_token_ids = self.tokenizer(
            [di.target_string for di in batch],
            add_special_tokens=False,
            max_length=self.decoder_sequence_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels = replace_pad_token_id_with_negative_hundred_for_hf_transformers_automatic_batch_transformation(
            self.tokenizer, target_token_ids
        )

        data_indices = [di.data_index for di in batch]

        return Batch(images=images, labels=labels, data_indices=data_indices)


def build_model(config: types.SimpleNamespace or object) -> Model:
    donut_processor = transformers.DonutProcessor.from_pretrained(
        config.pretrained_model_name
    )
    donut_processor.image_processor.size = dict(
        width=config.image_width, height=config.image_height
    )
    donut_processor.image_processor.do_align_long_axis = False

    tokenizer = donut_processor.tokenizer

    encoder_decoder_config = transformers.VisionEncoderDecoderConfig.from_pretrained(
        config.pretrained_model_name
    )
    encoder_decoder_config.encoder.image_size = (
        config.image_width,
        config.image_height,
    )

    encoder_decoder = transformers.VisionEncoderDecoderModel.from_pretrained(
        config.pretrained_model_name, config=encoder_decoder_config
    )
    encoder_decoder_config.pad_token_id = tokenizer.pad_token_id
    encoder_decoder_config.decoder_start_token_id = (
        tokenizer.convert_tokens_to_ids(
            get_extra_tokens().benetech_prompt
        )
    )
    encoder_decoder_config.bos_token_id = encoder_decoder_config.decoder_start_token_id
    encoder_decoder_config.eos_token_id = (
        tokenizer.convert_tokens_to_ids(
            get_extra_tokens().benetech_prompt_end
        )
    )

    extra_tokens = list(get_extra_tokens().__dict__.values())
    add_unknown_tokens_to_tokenizer(tokenizer, encoder_decoder, extra_tokens)
    unknown_dataset_tokens = load_pickle_or_build_object_and_save(
        config.unknown_tokens_for_tokenizer_path,
        lambda: list(find_unknown_tokens_for_tokenizer(tokenizer).keys()),
    )
    add_unknown_tokens_to_tokenizer(tokenizer, encoder_decoder, unknown_dataset_tokens)
    tokenizer.eos_token_id = encoder_decoder_config.eos_token_id

    batch_collate_function = BatchCollateFunction(
        processor=donut_processor,
        tokenizer=tokenizer,
        decoder_sequence_max_length=config.decoder_sequence_max_length,
    )

    return Model(
        processor=donut_processor,
        tokenizer=tokenizer,
        encoder_decoder=encoder_decoder,
        batch_collate_function=batch_collate_function,
        config=config,
    )


def generate_token_strings(
        model: Model, images: torch.Tensor, skip_special_tokens=True
) -> list[str]:
    decoder_output = model.encoder_decoder.generate(
        images,
        max_length=10
        if model.config.debug
        else model.config.decoder_sequence_max_length,
        eos_token_id=model.tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    return model.tokenizer.batch_decode(
        decoder_output.sequences, skip_special_tokens=skip_special_tokens
    )


def predict_string(model: Model, image):
    image = model.processor(
        image, random_padding=False, return_tensors="pt"
    ).pixel_values
    string = generate_token_strings(model, image)[0]
    return string


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(config)
        self.encoder_decoder = self.model.encoder_decoder

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int, dataset_idx: int = 0):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss)

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        outputs = self.encoder_decoder(
            pixel_values=batch.images, labels=batch.labels
        )
        loss = outputs.loss
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams["config"].learning_rate
        )
        return optimizer
