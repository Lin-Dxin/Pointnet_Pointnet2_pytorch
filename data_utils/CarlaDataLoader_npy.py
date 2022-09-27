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

    def __init__(self, carla_dir='data/carla', transform=None, split='train', proportion=[0.8, 0.1, 0.1],
                 label_weights=np.random.normal(size=23), sample_rate=0.1, numpoints=1024 * 8, need_speed=True,
                 block_size=1.0):
        self.split = split
        self.proportion = proportion
        # rootpath = os.path.abspath('..')
        self.block_size = block_size
        self.carla_dir = os.path.join(carla_dir)
        self.transform = transform
        self.label_weights = label_weights
        self.numpoints = numpoints
        all_file = os.listdir(self.carla_dir)
        self.need_speed = need_speed
        datanum = len(all_file)
        train_offset = int(datanum * proportion[0])
        test_offset = int(datanum * proportion[1]) + train_offset
        eval_offset = int(datanum * proportion[2]) + test_offset
        if split == 'train':
            print('Train Scene Data Loading..')
            all_file = all_file[:train_offset]
        if split == 'test':
            print('Test Scene Data Loading..')
            all_file = all_file[train_offset:test_offset]
        if split == 'eval':
            print('Eval Scene Data Loading..')
            all_file = all_file[test_offset:eval_offset]

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
            if self.need_speed:
                temp = [_raw[0], _raw[1], _raw[2], _raw[3]]
            else:
                temp = [_raw[0], _raw[1], _raw[2]]
            point.append(temp)
            label.append(_raw[5])
        point = np.asarray(point)
        label = np.asarray(label)
        N_points = len(label)
        cnt = 0
        while True:
            center = point[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (point[:, 0] >= block_min[0]) & (point[:, 0] <= block_max[0]) & (point[:, 1] >= block_min[1]) & (
                        point[:, 1] <= block_max[1]))[0]
            # print(point_idxs.size)
            cnt += 1
            # print(cnt)
            if point_idxs.size > 1024 or cnt > 100:
                # print("success! with cnt:")
                # print(cnt)
                break
        if point_idxs.size >= self.numpoints:
            selected_point_idxs = np.random.choice(point_idxs, self.numpoints, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.numpoints, replace=True)
        selected_data = np.array(point, dtype=np.float32)[selected_point_idxs]
        selected_label = np.array(label, dtype=np.float32)[selected_point_idxs]
        return selected_data, selected_label

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    point_data = CarlaDataset(carla_dir='../data/carla', split='test', need_speed=False)
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
