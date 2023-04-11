from pathlib import Path
import torch
from .coco_style_transfer import CocoStyleTransfer, std, mean

def denorm(tensor, device):
    std_ = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean_ = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std_ + mean_, 0, 1)
    return res


def build_dataset(image_set, args):
    content_folder = Path(args.content_folder)
    style_folder = Path(args.style_folder)
    print(f"Loading {image_set}...")
    assert content_folder.exists(), f'provided COCO path {content_folder} does not exist'
    assert style_folder.exists(), f'provided wikiart path {style_folder} does not exist'

    dataset = CocoStyleTransfer(content_folder, style_folder,args.img_size)
    return dataset
