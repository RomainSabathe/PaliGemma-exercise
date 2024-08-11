import torch

from paligemma.siglip import ViT, ViTConfiguration, break_images_into_patches

batch_size = 2
height = 224
width = 224
channels = 3


def test_initialize_vit():
    conf = ViTConfiguration()
    model = ViT(conf)


def test_inference_vit():
    conf = ViTConfiguration()
    model = ViT(conf)

    imgs = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    y = model(imgs)


def test_initialize_patch_embedder():
    patch_embedder = PatchEmbedder()


def test_break_images_into_patches_generic():
    # TODO: add cases where the user provides tuples, as well as patch sizes that do not
    # divide the image size.
    imgs = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    patches = break_images_into_patches(imgs, patch_size=16)

    assert len(patches.shape) == 4
    n_patches, patch_channels, patch_height, patch_width = patches.shape

    assert (
        n_patches == 14 * 14 * 2
    )  # horizontal patches * vertical patches * batch_size
    assert patch_height == 16
    assert patch_width == 16
    assert patch_channels == channels

    assert patches.dtype == torch.float32


def test_break_images_into_patches_mini_usecase():
    imgs = torch.arange(16).view(1, 1, 4, 4)
    patches = break_images_into_patches(imgs, patch_size=2)

    assert torch.equal(patches[0], torch.Tensor([[[0, 1], [4, 5]]]))
    assert torch.equal(patches[1], torch.Tensor([[[2, 3], [6, 7]]]))
    assert torch.equal(patches[2], torch.Tensor([[[8, 9], [12, 13]]]))
    assert torch.equal(patches[3], torch.Tensor([[[10, 11], [14, 15]]]))
