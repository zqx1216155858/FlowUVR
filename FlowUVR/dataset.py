import torch
import os
from glob import glob
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import random
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
    def __init__(self, dataset_folder, split, image_prep, tokenizer, num_frames=1,samples_per_video=10):
        """
        A dataset class for loading unpaired video data from the same videos but different frames,
        typically used in unsupervised learning tasks like video-to-video translation.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): The image preprocessing transformation to apply to each frame.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        - num_frames (int): Number of consecutive frames to load for each video clip.
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")

        self.tokenizer = tokenizer
        self.num_frames = num_frames

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

        # Find all video folders in source and target directories
        self.video_folders_src = sorted([f for f in os.listdir(self.source_folder)
                                         if os.path.isdir(os.path.join(self.source_folder, f))])
        self.video_folders_tgt = sorted([f for f in os.listdir(self.target_folder)
                                         if os.path.isdir(os.path.join(self.target_folder, f))])

        self.samples_per_video = samples_per_video


        common_videos = set(self.video_folders_src) & set(self.video_folders_tgt)
        self.common_videos = sorted(list(common_videos))

        self.total_samples = len(self.common_videos) * samples_per_video
        # Precompute frame lists for each video
        self.video_frame_lists_src = {}
        self.video_frame_lists_tgt = {}

        for video_folder in self.common_videos:

            video_path_src = os.path.join(self.source_folder, video_folder)
            frames_src = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
                frames_src.extend(sorted(glob(os.path.join(video_path_src, ext))))
            self.video_frame_lists_src[video_folder] = frames_src

            video_path_tgt = os.path.join(self.target_folder, video_folder)
            frames_tgt = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
                frames_tgt.extend(sorted(glob(os.path.join(video_path_tgt, ext))))
            self.video_frame_lists_tgt[video_folder] = frames_tgt

        # Filter out videos that don't have enough frames in both domains
        self.valid_videos = [vid for vid in self.common_videos
                             if len(self.video_frame_lists_src[vid]) >= num_frames and
                             len(self.video_frame_lists_tgt[vid]) >= num_frames]

        if len(self.valid_videos) == 0:
            raise ValueError(f"dont find {num_frames} video")

        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of samples in the dataset.
        """
        return self.total_samples

    def _load_video_frames(self, video_folder, frame_list, is_source=True):
        """Load consecutive frames from a video folder."""
        if len(frame_list) < self.num_frames:
            raise ValueError(
                f"Video {video_folder} has only {len(frame_list)} frames, but {self.num_frames} are required")

        # Randomly select a starting frame index
        max_start_idx = len(frame_list) - self.num_frames
        start_idx = random.randint(0, max_start_idx)

        frames = []
        for i in range(start_idx, start_idx + self.num_frames):
            img_path = frame_list[i]
            img_pil = Image.open(img_path).convert("RGB")
            img_t = F.to_tensor(self.T(img_pil))
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            frames.append(img_t)

        # Stack frames to create (T, C, H, W) tensor
        video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
        return video_tensor

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned video clips from the same video but different random frames,
        along with their corresponding tokenized captions.

        Parameters:
        - index (int): The index of the video to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source video clip of shape (T, C, H, W)
            - "pixel_values_tgt": The processed target video clip of shape (T, C, H, W)
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        video_idx_src = index // self.samples_per_video
        video_folder_src = self.valid_videos[video_idx_src]
        frame_list_src = self.video_frame_lists_src[video_folder_src]
        video_folder_tgt = random.choice(self.valid_videos)
        if len(self.valid_videos) > 1:
            while video_folder_tgt == video_folder_src:
                video_folder_tgt = random.choice(self.valid_videos)

        frame_list_tgt = self.video_frame_lists_tgt[video_folder_tgt]

        random.seed(index + video_idx_src * 1000)

        video_src = self._load_video_frames(
            video_folder_src, frame_list_src, is_source=True
        )
        video_tgt = self._load_video_frames(
            video_folder_tgt, frame_list_tgt, is_source=False
        )

        return {
            "pixel_values_src": video_src,  # (T, C, H, W)
            "pixel_values_tgt": video_tgt,  # (T, C, H, W)
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }

