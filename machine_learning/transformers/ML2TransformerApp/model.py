from einops.layers.torch import Rearrange
import einops
import math
import pytorch_lightning as pl
import torch.nn as nn
import torch


class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_len=5000):
        super().__init__()

        positions = torch.arange(max_sequence_len)
        even_embedding_indices = torch.arange(0, d_model, 2)

        expression = torch.exp(even_embedding_indices * (-math.log(10000.0) / d_model))
        expression = torch.einsum("i, j -> ij", positions, expression)

        even_encodings = torch.sin(expression)
        odd_encodings = torch.cos(expression)

        positional_encodings = einops.rearrange(
            [even_encodings, odd_encodings],
            "even_odd pos embed -> pos (embed even_odd)",
        )

        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, batch):
        seq_len = batch.size(1)
        positional_encodings = self.positional_encodings[:seq_len, :]
        return batch + positional_encodings


class ImageEmbedding(nn.Module):
    """Reshape image into patches and project into given dimension"""

    def __init__(self, d_model, input_width, input_height, patch_size, dropout):
        super().__init__()
        assert (
            input_width % patch_size == 0 and input_height % patch_size == 0
        ), "Cannot split image in patches"
        tokenize = Rearrange(
            "b c (h1 h2) (w1 w2) -> b (c h1 w1) (h2 w2)", h2=patch_size, w2=patch_size
        )
        project = nn.Linear(patch_size**2, d_model)
        self.embed = nn.Sequential(
            tokenize, project, AddPositionalEncoding(d_model), nn.Dropout(p=dropout)
        )

    def forward(self, image_batch):
        image_batch = self.embed(image_batch)
        return image_batch


class TexEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.add_positional_encoding = AddPositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, tex_ids_batch):
        tex_ids_batch = self.embedding(tex_ids_batch.long()) * math.sqrt(self.d_model)
        tex_ids_batch = self.add_positional_encoding(tex_ids_batch)
        tex_ids_batch = self.dropout(tex_ids_batch)
        return tex_ids_batch


class Transformer(pl.LightningModule):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        nhead: int,
        image_width: int,
        image_height: int,
        tgt_vocab_size: int,
        pad_idx: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.d_model = d_model
        self.src_tok_emb = ImageEmbedding(
            d_model, image_width, image_height, patch_size=16, dropout=dropout
        )
        self.tgt_tok_emb = TexEmbedding(d_model, tgt_vocab_size, dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.tgt_tok_emb.embedding.weight = self.generator.weight
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=pad_idx, label_smoothing=0.1
        )
        self.save_hyperparameters()

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        """The positions of masks with ``True``
            are not allowed to attend while ``False`` values will be unchanged.
        The positions of padding masks with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged."""
        src = self.src_tok_emb(src)
        tgt = self.tgt_tok_emb(tgt)
        outs = self.transformer(
            src,
            tgt,
            src_mask,
            tgt_mask,
            memory_mask,
            src_padding_mask,
            tgt_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_mask=None, src_padding_mask=None):
        src = self.src_tok_emb(src)
        return self.transformer.encoder(src, src_mask, src_padding_mask)

    def decode(
        self, tgt, memory=None, tgt_mask=None, memory_mask=None, tgt_padding_mask=None
    ):
        tgt = self.tgt_tok_emb(tgt)
        outs = self.transformer.decoder(
            tgt, memory, tgt_mask, memory_mask, tgt_padding_mask
        )
        return self.generator(outs)

    def _shared_step(self, batch):
        src = batch["images"]
        tgt = batch["tex_ids"]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask = None
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_input.shape[1]
        ).to(self.device, torch.bool)
        memory_mask = None
        src_padding_mask = None
        tgt_padding_mask = torch.logical_not(batch["tex_attention_masks"][:, :-1])

        outs = self(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            memory_mask,
            src_padding_mask,
            tgt_padding_mask,
        )
        loss = self.loss_fn(
            einops.rearrange(outs, "b n prob -> b prob n"), tgt_output.long()
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, NoamLRLambda(self.d_model)
        )
        return [optimizer], [scheduler]


class NoamLRLambda:
    def __init__(self, d_model, factor=1, warmup=4000):
        """
        :param d_model: size of hidden model dimension
        :param factor: multiplicative factor
        :param warmup: number of warmup steps
        """
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup

    def __call__(self, step):
        step += 1
        return (
            self.factor
            * self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
