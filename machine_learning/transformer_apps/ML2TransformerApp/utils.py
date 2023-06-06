from constants import TOKENIZER_PATH

import einops
import os
import random
from pytorch_lightning.callbacks import Callback
import torch
import torch.nn.functional as F
from torchvision import transforms


class LogImageTexCallback(Callback):
    def __init__(self, logger, top_k, max_length):
        self.logger = logger
        self.top_k = top_k
        self.max_length = max_length
        self.tex_tokenizer = torch.load(TOKENIZER_PATH)
        self.tensor_to_PIL = transforms.ToPILImage()

    def on_validation_batch_start(
        self, trainer, transformer, batch, batch_idx, dataloader_idx
    ):
        if batch_idx != 0 or dataloader_idx != 0:
            return
        sample_id = random.randint(0, len(batch["images"]) - 1)
        image = batch["images"][sample_id]
        texs_predicted = beam_search_decode(
            transformer, image, top_k=self.top_k, max_length=self.max_length
        )
        image = self.tensor_to_PIL(image)
        tex_true = self.tex_tokenizer.decode(
            list(batch["tex_ids"][sample_id].to("cpu", torch.int))
        )
        self.logger.log_image(
            key="samples",
            images=[image],
            caption=[f"True: {tex_true}\nPredicted: " + "\n".join(texs_predicted)],
        )


@torch.inference_mode()
def beam_search_decode(
    transformer, image, image_transform=None, top_k=10, max_length=100
):
    """Performs decoding maintaining k best candidates"""

    def get_tgt_padding_mask(tgt):
        mask = tgt == tex_tokenizer.token_to_id("[SEP]")
        mask = torch.cumsum(mask, dim=1)
        mask = mask.to(transformer.device, torch.bool)
        return mask

    if image_transform:
        image = image_transform(image)

    assert (
        torch.is_tensor(image) and len(image.shape) == 3
    ), "Image must be a 3 dimensional tensor (c h w)"
    src = einops.rearrange(image, "c h w -> () c h w").to(transformer.device)
    memory = transformer.encode(src)

    tex_tokenizer = torch.load(TOKENIZER_PATH)
    candidates_tex_ids = [[tex_tokenizer.token_to_id("[CLS]")]]
    candidates_log_prob = torch.tensor(
        [0], dtype=torch.float, device=transformer.device
    )

    while (
        candidates_tex_ids[0][-1] != tex_tokenizer.token_to_id("[SEP]")
        and len(candidates_tex_ids[0]) < max_length
    ):
        candidates_tex_ids = torch.tensor(
            candidates_tex_ids, dtype=torch.float, device=transformer.device
        )
        tgt_mask = transformer.transformer.generate_square_subsequent_mask(
            candidates_tex_ids.shape[1]
        ).to(transformer.device, torch.bool)
        shared_memories = einops.repeat(
            memory, f"one n d_model -> ({candidates_tex_ids.shape[0]} one) n d_model"
        )
        outs = transformer.decode(
            tgt=candidates_tex_ids,
            memory=shared_memories,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_padding_mask=get_tgt_padding_mask(candidates_tex_ids),
        )
        outs = einops.rearrange(outs, "b n prob -> b prob n")[:, :, -1]
        vocab_size = outs.shape[1]
        outs = F.log_softmax(outs, dim=1)
        outs += einops.rearrange(candidates_log_prob, "prob -> prob ()")
        outs = einops.rearrange(outs, "b prob -> (b prob)")
        candidates_log_prob, indices = torch.topk(outs, k=top_k)

        new_candidates = []
        for index in indices:
            candidate_id, token_id = divmod(index.item(), vocab_size)
            new_candidates.append(
                candidates_tex_ids[candidate_id].to(int).tolist() + [token_id]
            )
        candidates_tex_ids = new_candidates

    candidates_tex_ids = torch.tensor(candidates_tex_ids)
    padding_mask = get_tgt_padding_mask(candidates_tex_ids).cpu()
    candidates_tex_ids = candidates_tex_ids.masked_fill(
        padding_mask & (candidates_tex_ids != tex_tokenizer.token_to_id("[SEP]")),
        tex_tokenizer.token_to_id("[PAD]"),
    ).tolist()
    texs = tex_tokenizer.decode_batch(candidates_tex_ids, skip_special_tokens=True)
    texs = [tex.replace("\\ ", "\\") for tex in texs]
    return texs


def average_checkpoints(model_type, checkpoints_dir):
    """Returns model averaged from checkpoints
    Args:
        :model_type: -- pytorch_lightning.LightningModule that corresponds to checkpoints
        :checkpoints_dir: -- path to checkpoints
    """
    checkpoints = [checkpoint.path for checkpoint in os.scandir(checkpoints_dir)]
    n_models = len(checkpoints)
    assert n_models > 0
    average_model = model_type.load_from_checkpoint(checkpoints[0])

    for checkpoint in checkpoints[1:]:
        model = model_type.load_from_checkpoint(checkpoint)
        for weight, weight_to_add in zip(
            average_model.parameters(), model.parameters()
        ):
            weight.data.add_(weight_to_add.data)

    for weight in average_model.parameters():
        weight.data.divide_(n_models)

    return average_model
