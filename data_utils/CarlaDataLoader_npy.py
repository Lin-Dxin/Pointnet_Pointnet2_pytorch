import os
import numpy as np
import torch
# from plyfile import PlyData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from time import time
import tqdm


class CarlaDataset(Dataset):
    label_weights = np.random.uniform(size=23)

    def __init__(self, carla_dir='data/carla', transform=None, split='train', proportion=0.8,
                 label_weights=np.random.normal(size=23), sample_rate=0.1, numpoints=8000, need_speed=True):
        self.split = split
        self.proportion = proportion
        # rootpath = os.path.abspath('..')
        self.carla_dir = os.path.join(carla_dir)
        self.transform = transform
        self.label_weights = label_weights
        self.numpoints = numpoints
        all_file = os.listdir(self.carla_dir)
        self.need_speed = need_speed
        datanum = len(all_file)
        offset = int(datanum * proportion)
        if split == 'train':
            all_file = all_file[:offset]
        if split == 'test':
            all_file = all_file[offset:]
        self.file_list = all_file
        self.file_len = len(all_file)
        # 只读取文件，不保存： 记录点数、初始化权重、标准化
        num_all_point = []
        for file_name in all_file:
            path = os.path.join(self.carla_dir, file_name)
            data = np.load(path)
            num_all_point.append(len(data))  # 记录点云数

        sample_prob = num_all_point / np.sum(num_all_point)
        num_iter = int(np.sum(num_all_point) * sample_rate / numpoints)
        room_idxs = []
        for index in range(self.file_len):
            room_idxs.extend([index] * int(round((sample_prob[index]) * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        roompath = os.path.join(self.carla_dir, self.file_list[room_idx])
        raw_data = np.load(roompath)
        point = []
        label = []
        for _raw in raw_data:
            if self.need_speed == True:
                temp = [_raw[0], _raw[1], _raw[2], _raw[3]]
            else:
                temp = [_raw[0], _raw[1], _raw[2]]
            point.append(temp)
            label.append(_raw[5])
        point = np.asarray(point)
        label = np.asarray(label)
        if label.size >= self.numpoints:
            selected_point_idxs = np.random.choice(label.size, self.numpoints, replace=False)
        else:
            selected_point_idxs = np.random.choice(label.size, self.numpoints, replace=True)
        selected_data = np.array(point, dtype=np.float32)[selected_point_idxs]
        selected_label = np.array(label, dtype=np.float32)[selected_point_idxs]
        return selected_data, selected_label

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    point_data = CarlaDataset(carla_dir='../data/carla', split='train')
    train_loader = DataLoader(point_data, batch_size=16, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True,
                              worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    for i, (input, target) in enumerate(train_loader):
        print(input.shape)
        print(target.shape)

        break
