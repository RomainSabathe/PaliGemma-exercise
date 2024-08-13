from dataclasses import dataclass

import einops
import torch
import torch.nn as nn


@dataclass
class ViTConfiguration:
    image_size: int = 224
    in_channels: int = 3
    hidden_dim: int = 1_024  # "latent vector size D" in the paper
    patch_size: int = 16
    num_layers: int = 3

    @property
    def num_patches(self) -> int:
        return self.image_size**2 // self.patch_size**2


@dataclass
class SigLipVisionConfiguration:
    encoder_configuration: ViTConfiguration


class SigLipVisionModel(nn.Module):
    def __init__(self, configuration: SigLipVisionConfiguration):
        super(SigLipVisionModel, self).__init__()
        self.configuration = configuration
        self.vision_encoder = ViT()


@dataclass
class ImageTokenizerConfiguration(nn.Module):
    in_channels: int = 3
    patch_size: int = 16
    embedding_size: int = 1_024


class ImageTokenizer(nn.Module):
    def __init__(self, configuration: ImageTokenizerConfiguration):
        super(ImageTokenizer, self).__init__()

        # Q: what is the relationship between the token dimensions and the hidden dimension?
        # A: they're the same. (ViT has only one hidden dim throughout)
        self.conv2d = nn.Conv2d(
            in_channels=configuration.in_channels,
            out_channels=configuration.embedding_size,
            kernel_size=configuration.patch_size,
            stride=configuration.patch_size,
            padding="valid",
        )

    def forward(self, imgs_bchw: torch.Tensor) -> torch.Tensor:
        # Returns shape [b, p, d] where d = embedding_size
        feature_maps_bdphpw = self.conv2d(
            imgs_bchw
        )  # where d = embedding dim and ph/pw = the number of patches in both directions
        return einops.rearrange(feature_maps_bdphpw, "b d ph pw -> b (ph pw) d")


def get_initial_positional_embeddings(
    num_patches: int, embedding_size: int
) -> torch.Tensor:
    # TODO last 15 minutes: check which initialisation is chosen
    return nn.Parameter(torch.randn(num_patches, embedding_size))


@dataclass
class TransformerEncoderConfiguration:
    sequence_length: int
    num_layers: int = 3
    hidden_dim: int = 728


class TransformerEncoder(nn.Module):
    def __init__(self, configuration: TransformerEncoderConfiguration):
        super(TransformerEncoder, self).__init__()
        self.configuration = configuration
        block_configuration = TransformerBlockConfiguration(
            sequence_length=configuration.sequence_length,
            hidden_dim=configuration.hidden_dim,
        )
        self.blocks = [
            TransformerBlock(block_configuration)
            for _ in range(configuration.num_layers)
        ]
        self.out_layer_norm = LayerNorm(
            sequence_length=configuration.sequence_length,
            hidden_dim=configuration.hidden_dim,
        )

    def forward(self, in_tokens_bnd: torch.Tensor) -> torch.Tensor:
        x = in_tokens_bnd
        for block in self.blocks:
            x = block(x)

        return self.out_layer_norm(x)


class LayerNorm(nn.Module):
    def __init__(self, sequence_length: int, hidden_dim: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.bias_nd = nn.Parameter(torch.zeros(sequence_length, hidden_dim))
        self.gain_nd = nn.Parameter(torch.ones(sequence_length, hidden_dim))

    def forward(self, x_bnd: torch.Tensor) -> torch.Tensor:
        mu_b11 = einops.reduce(x_bnd, "b n d -> b () ()", "mean")
        variance_b11 = einops.reduce(
            (x_bnd - mu_b11) ** 2.0, "b n d -> b () ()", "mean"
        )
        sigma_b11 = torch.sqrt(variance_b11 + self.eps)

        x_bnd = (x_bnd / mu_b11) / sigma_b11
        x_bnd = (x_bnd + self.bias_nd) * self.gain_nd
        return x_bnd


@dataclass
class TransformerBlockConfiguration:
    sequence_length: int
    hidden_dim: int


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, k_dim: int | None = None, v_dim: int |None  = None):
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.k_dim = k_dim if k_dim is not None else hidden_dim
        self.v_dim = v_dim if v_dim is not None else hidden_dim

        (QK) V

        # Hello! If you're back, know that we are trying to figure out an efficient way of
        # implementing multihead attention. Should we have independent Attention modules, or
        # is there a way of batching everything under a single operation? For instance via a
        # (num_heads out_feat in_feat) weight matrix.
        self.k_projectors = nn.Linear(in_features=hidden_dim, out_features=k_dim)
        self.k_projectors = 
        self.

    def forward(self, x_bnd: torch.Tensor) -> torch.Tensor:

        


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

    def forward(self, x_bnd: torch.Tensor) -> torch.Tensor:
        pass


class TransformerBlock(nn.Module):
    def __init__(self, configuration: TransformerBlockConfiguration):
        super(TransformerBlock, self).__init__()
        self.configuration = configuration

        self.layer_norm_1 = LayerNorm(
            sequence_length=configuration.sequence_length,
            hidden_dim=configuration.hidden_dim,
        )
        self.layer_norm_2 = LayerNorm(
            sequence_length=configuration.sequence_length,
            hidden_dim=configuration.hidden_dim,
        )
        self.multihead_attention = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x_bnd: torch.Tensor) -> torch.Tensor:
        x_skip_bnd = x_bnd
        x_bnd = self.layer_norm_1(x_bnd)
        x_bnd = self.multihead_attention(x_bnd)
        x_bnd += x_skip_bnd

        x_skip_bnd = x_bnd
        x_bnd = self.layer_norm_2(x_bnd)
        x_bnd = self.mlp(x_bnd)
        x_bnd += x_skip_bnd

        return x_bnd


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()


class ViT(nn.Module):
    def __init__(self, configuration: ViTConfiguration):
        super(ViT, self).__init__()
        self.configuration = configuration

        image_tokenizer_configuration = ImageTokenizerConfiguration(
            in_channels=configuration.in_channels,
            embedding_size=configuration.hidden_dim,
            patch_size=configuration.patch_size,
        )
        self.image_tokenizer = ImageTokenizer(image_tokenizer_configuration)
        # Q: is this really the right way of writing/initialising the positional embedding?
        self.positional_embeddings = get_initial_positional_embeddings(
            num_patches=configuration.num_patches,
            embedding_size=configuration.hidden_dim,
        )
        encoder_configuration = TransformerEncoderConfiguration(
            sequence_length=configuration.num_patches,
            num_layers=configuration.num_layers,
            hidden_dim=configuration.hidden_dim,
        )
        self.encoder = TransformerEncoder(encoder_configuration)
        self.classification_head = ClassificationHead()

    def forward(
        self, imgs_pixels_bchw: torch.Tensor
    ) -> torch.Tensor:  # img_pixels or imgs_pixels ?
        imgs_tokens_nd = self.image_tokenizer(imgs_pixels_bchw)
        imgs_tokens_nd += self.positional_embeddings

        # TODO: add the special classification token

        output_tokens_nd = self.encoder(imgs_tokens_nd)

        # Q: what is this module supposed to return? Just the transformation of the classification token?
        #    the list of output embeddings?
        return output_tokens_nd[
            :, 0
        ]  # returning the classification token for all items in the batch.
