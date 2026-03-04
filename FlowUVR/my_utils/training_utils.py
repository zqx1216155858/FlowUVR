import os
import random
import argparse
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import glob


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_cycle", default=10, type=float)
    parser.add_argument("--lambda_cycle_ms_ssim", type=float, default=1)
    parser.add_argument("--lambda_temp_cyc", type=float, default=0.5)
    parser.add_argument("--lambda_temp_gen", type=float, default=0.5)
    parser.add_argument("--lambda_fm", type=float, default=10)
    parser.add_argument("--lambda_ke", type=float, default=1)
    parser.add_argument("--lambda_div", type=float, default=1)


    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", default="dataset", type=str)
    parser.add_argument("--train_img_prep",default="crop_128")
    parser.add_argument("--val_img_prep", default="no_resize")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--max_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=300000)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="gparmar_unpaired_s2c_cycle_debug_v2")
    parser.add_argument("--validation_steps", type=int, default=5000,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=1e-5,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)



    args = parser.parse_args()
    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_256":
        T = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(256),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["crop_256"]:
        T = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["crop_128"]:
        T = transforms.Compose([
            transforms.RandomCrop((128, 128)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T



class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }


def pad_to_multiple(tensor, multiple=16):

    if len(tensor.shape) == 3:  # (C, H, W)
        h, w = tensor.shape[1], tensor.shape[2]
    else:  # (B, C, H, W)
        h, w = tensor.shape[2], tensor.shape[3]

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if len(tensor.shape) == 3:
        padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    return padded, (h, w, pad_h, pad_w)


def crop_to_original(tensor, original_shape):

    h, w, pad_h, pad_w = original_shape
    if len(tensor.shape) == 3:  # (C, H, W)
        return tensor[:, :h, :w]
    else:  # (B, C, H, W)
        return tensor[:, :, :h, :w]

def get_all_test_images(test_folder):
    all_images = []
    video_folders = [f for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, f))]

    for video_folder in video_folders:
        video_path = os.path.join(test_folder, video_folder)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            all_images.extend(glob.glob(os.path.join(video_path, ext)))

    return sorted(all_images)


from pytorch_msssim import MS_SSIM
import torch
import numpy as np
def divergence_hutchinson_fd(forward_func, x, eps=1e-3):
    e = torch.randn_like(x)
    e = e / (e.norm() + 1e-8)  # normalize

    v = forward_func(x)
    v_eps = forward_func(x + eps * e)

    div = ((v_eps - v) * e).sum() / eps  # now stable!

    return div

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):

        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass


        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class MS_SSIM_Loss(MS_SSIM):
    def __init__(self, data_range=2.0, **kwargs):
        weights_4_scales = [0.0448, 0.2856, 0.3001, 0.2363]

        weights_sum = sum(weights_4_scales)
        normalized_weights = [w / weights_sum for w in weights_4_scales]

        super().__init__(
            data_range=data_range,
            size_average=True,
            weights=normalized_weights,
            channel=3
        )

    def forward(self, img1, img2):
        return 1 - super().forward(img1, img2)

    def get_min_image_size(self):
        """返回所需的最小图像尺寸"""
        return (self.win_size - 1) * 8
