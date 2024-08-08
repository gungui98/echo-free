import os.path
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import einops
import imageio
import torch
from torchvision import transforms as T, utils
from torch.utils.data import Dataset
import torch.nn.functional as F


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class DummyDataset(Dataset):
    def __init__(self, image_size, channels=3, num_frames=16, num_samples=1):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.num_classes = 4
        self.images = torch.randn(num_samples, self.channels, self.num_frames, self.image_size, self.image_size)
        self.seg_maps = torch.randint(0, self.num_classes, (num_samples, self.num_frames, self.image_size, self.image_size))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "cond": self.seg_maps[index]
        }


class EchoVideoDataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            channels=3,
            num_frames=16,
            split="train",
            mask_condition=False,
    ):
        super().__init__()
        self.folder = folder if isinstance(folder, Path) else Path(folder)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.channels = channels
        self.num_frames = num_frames
        data_folders = open(self.folder / f"{split}.txt").read().splitlines()
        cache_file = os.path.join(self.folder, f"cache_data_{image_size}_{split}.pth")
        self.images = self._get_video_data(self.folder / 'images')
        self.seg_maps = self._get_video_data(self.folder / 'segmaps')
        # get only the data in the data_folders
        self.images = {k: v for k, v in self.images.items() if k in data_folders}
        self.seg_maps = {k: v for k, v in self.seg_maps.items() if k in data_folders}
        self.mask_condition = mask_condition
        # check if the dataset is echonet, since we don't have the cone mask for echonet
        if "echonet" in str(self.folder):
            self.is_echonet = True
        else:
            self.is_echonet = False

        if os.path.exists(cache_file):
            self.images_tensor, self.seg_maps_tensor = torch.load(cache_file)
            print("load data from cache", cache_file)
        else:
            self.images_tensor = []
            self.seg_maps_tensor = []
            self.load_to_memory()
            torch.save((self.images_tensor, self.seg_maps_tensor), cache_file)
            print("save data to cache", cache_file)

    def _get_video_data(self, path):
        """get data from directory, the data has following structure
            --data
                --images
                    --patient0001
                        --0001.png
                        --0002.png
                        ...
                    --patient0002
                        --0001.png
                        --0002.png
                        ...
                --segmaps
                    --patient0001
                        --0001.png
                        --0002.png
                        ...
        """
        if isinstance(path, str):
            path = Path(path)
        data = defaultdict(dict)
        for p in path.glob('**/*'):
            if p.is_file():
                patient_id, frame_id = p.parent.name, p.stem
                data[patient_id][frame_id] = p
        return data

    def __len__(self):
        return len(self.images)

    def load_to_memory(self):
        """
        load data to memory, so that it can be accessed faster
        """
        for index in range(len(self)):
            patient_id = list(self.images.keys())[index]
            frames = list(self.images[patient_id].values())
            seg_maps = list(self.seg_maps[patient_id].values())

            frames = sorted(frames, key=lambda x: int(x.stem))
            seg_maps = sorted(seg_maps, key=lambda x: int(x.stem))
            frames = [imageio.imread(frame, mode='L') for frame in frames]
            seg_maps = [imageio.imread(seg_map) for seg_map in seg_maps]
            frames = torch.tensor(np.array(frames))
            seg_maps = torch.tensor(np.array(seg_maps))
            # resize the frames
            frames = T.Resize(self.image_size)(frames)
            seg_maps = T.Resize(self.image_size,
                                interpolation=T.InterpolationMode.NEAREST)(seg_maps)
            # duplicate the frames to RGB channels
            frames = einops.repeat(frames, 't h w -> c t h w', c=self.channels)
            seg_maps = einops.rearrange(seg_maps, 't h w -> 1 t h w')

            self.images_tensor.append(frames/ 255.0)
            self.seg_maps_tensor.append(seg_maps)

    def __getitem__(self, index):
        # read the data from self.data, convert to tensor
        frames = self.images_tensor[index]
        seg_maps = self.seg_maps_tensor[index]
        
        # if mask condition, then set all the pixels that are not background to 1
        if self.mask_condition:
            if self.is_echonet:
                # for echonet, we don't have the cone mask, so we set all pixels to zero
                seg_maps = torch.zeros_like(seg_maps)
            else:
                seg_maps[seg_maps >= 1] = 1

        frames = cast_num_frames(frames, frames=self.num_frames).float()
        seg_maps = cast_num_frames(seg_maps, frames=self.num_frames)[0].long()

        return {
            "image": frames,
            "cond": seg_maps
        }

if __name__ == "__main__":
    dataset =  EchoVideoDataset(
        folder="./data_camus_merge",
        image_size=128,
        channels=3,
        num_frames=16,
        split="train",
    )
    print(len(dataset))