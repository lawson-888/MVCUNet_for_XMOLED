import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.ndimage import rotate

def augment_numpy_image(numpy_image, flip_prob=0.5, rotation_angle=90):

    if np.random.rand() < flip_prob:
        numpy_image = np.flip(numpy_image, axis=1) 
    if np.random.rand() < flip_prob:
        angle = np.random.uniform(-rotation_angle, rotation_angle)
        numpy_image = rotate(numpy_image, angle, reshape=False, mode='constant', order=0, axes=(0, 1)) 

    return numpy_image


def data2tensor(file_name, pha, fre, transform_operator, map_type, input_ch, is_aug):
    raw_data = np.fromfile(file_name, dtype=np.float32)
    raw_data = np.reshape(raw_data, [9, fre, pha])
    raw_data = np.transpose(raw_data, [2, 1, 0])
    if is_aug:
        raw_data_aug = augment_numpy_image(raw_data)
    else:
        raw_data_aug = raw_data
    raw_data_tensor = transform_operator(raw_data_aug.copy())
    final_input_data = raw_data_tensor[input_ch, :, :]

    final_label_data = torch.unsqueeze(raw_data_tensor[map_type, :, :], dim=0)
    return final_input_data, final_label_data


class mapping_dataset(Dataset):
    def __init__(self, root_dir, transform, pha, fre, map_mission, input_ch, is_aug):
        self.root_dir = root_dir
        self.transform = transform
        self.data_name = os.listdir(self.root_dir)
        self.pha = pha
        self.fre = fre
        self.map = map_mission
        self.inputch = input_ch
        self.isaug = is_aug

    def __getitem__(self, idx):
        file_name = self.data_name[idx]
        fullfile_name = os.path.join(self.root_dir, file_name)
        (input_data, label_data) = data2tensor(fullfile_name, self.pha, self.fre, self.transform, self.map,
                                               self.inputch, self.isaug)
        return input_data, label_data

    def __len__(self):
        return len(self.data_name)

