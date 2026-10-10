import os
import joblib
import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from timm.data import create_transform
#from timm.data.transforms import _pil_interp


def build_transform(config, is_train=True):
    if is_train:
        transform = create_transform(
            input_size=(config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
            is_training=True,
            hflip=config.AUG.HFLIP,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation=config.DATA.INTERPOLATION,
        )
        transform.transforms[0] = transforms.Resize((config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
                                                    interpolation=_pil_interp(config.DATA.INTERPOLATION))
        return transform
    else:
        t = [
                transforms.Resize((config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
                               interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        return transforms.Compose(t)


class Chunked_sample_dataset(Dataset):
    def __init__(self, config, chunked_folder, last_flow=False):
        super(Chunked_sample_dataset, self).__init__()
        self.last_flow = last_flow

        chunked_files = sorted(os.listdir(chunked_folder))

        chunked_appearance, chunked_motion, chunked_bbox, chunked_pred_frame, chunked_id = [], [], [], [], []
        for chunked_file in chunked_files:
            chunked_samples = joblib.load(os.path.join(chunked_folder, chunked_file))
            chunked_appearance.append(chunked_samples["appearance"])
            chunked_motion.append(chunked_samples["motion"])
            chunked_bbox.append(chunked_samples["bbox"])
            chunked_pred_frame.append(chunked_samples["pred_frame"])
            chunked_id.append(chunked_samples["sample_id"])

        self.chunked_samples_appearance = np.concatenate([appearance for appearance in chunked_appearance], axis=0)
        self.chunked_samples_motion = np.concatenate([motion for motion in chunked_motion])
        self.chunked_samples_bbox = np.concatenate([bbox for bbox in chunked_bbox])
        self.chunked_samples_pred_frame = np.concatenate([pre for pre in chunked_pred_frame])
        self.chunked_samples_id = np.concatenate([id for id in chunked_id])

        # self.transform = build_transform(config=config, is_train=True)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])    # replace Transform
        self.flow_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.chunked_samples_id)

    def __getitem__(self, indice):
        appearance = self.chunked_samples_appearance[indice]
        motion = self.chunked_samples_motion[indice]
        bbox = self.chunked_samples_bbox[indice]
        pred_frame = self.chunked_samples_pred_frame[indice]

        # [#frame,h,w,c] to [h,w,#frame,c]
        x = np.transpose(appearance, [1, 2, 0, 3])
        x = np.reshape(x, (x.shape[0], x.shape[1], -1))
        frames = self.transform(x)
        frames = torch.split(frames, 3, dim=0)

        y = motion[1:] if not self.last_flow else motion[-1:]
        y = np.transpose(y, [1, 2, 0, 3])
        y = np.reshape(y, (y.shape[0], y.shape[1], -1))
        flow = self.flow_transform(y)

        return frames, flow, bbox.astype(np.float32), pred_frame, indice


def chunked_sample_loader(config, is_train=True):
    if is_train:
        chunked_folder = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, "training", config.DATA.CHUNKED_FOLDER)
    else:
        chunked_folder = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, "testing",
                                      config.DATA.CHUNKED_FOLDER)
    dataset = Chunked_sample_dataset(config=config, chunked_folder=chunked_folder, last_flow=True)

    if is_train:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.DATA.BATCH_SIZE * len(config.GPUS),
            shuffle=True,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

    return dataloader

