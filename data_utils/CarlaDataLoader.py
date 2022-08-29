import os
import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from time import time


class CarlaDataset(Dataset):
    label_weights = np.random.uniform(size=24)

    def __init__(self, carla_dir='data\\carla', transform=None, split='train', proportion=0.8,
                 label_weights=np.random.uniform(size=24), numpoints=4096):
        self.split = split
        self.proportion = proportion
        self.carla_dir = carla_dir
        self.transform = transform
        self.label_weights = label_weights
        self.numpoints = numpoints
        all_file = os.listdir(self.carla_dir)
        datanum = len(all_file)
        offset = int(datanum * proportion)
        if split == 'train':
            all_file = all_file[:offset]
        if split == 'test':
            all_file = all_file[offset:]
        data = []
        label = []

        print(type(data))
        for file in all_file:
            plydata = PlyData.read(os.path.join(self.carla_dir, file))
            raw_data = plydata.elements[0].data[:8000]
            for point in raw_data:
                temp = [point[0], point[1], point[2], point[3]]
                # print(temp)
                data.append(temp)
                # np.append(data, temp)
                # print(data)
                label.append(point[5])
                # np.append(label, point[5])
        n = 8000

        data = [data[j:j + n] for j in range(0, len(label), n)]
        label = [label[j:j + n] for j in range(0, len(label), n)]
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        self.data_len = len(label)
        self.point_label = label
        self.point_data = data
        print(self.point_label.shape)
        print(self.point_data.shape)
        # labelweights = np.zeros(23)
        # labelweights = labelweights.astype(np.float32)
        # labelweights = labelweights / np.sum(labelweights)
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)


    def __getitem__(self, idx):
        # selected_data = np.zeros((self.numpoints, 4))
        # selected_label = np.zeros()
        point = self.point_data[idx]
        label = self.point_label[idx]
        if len(label) >= self.numpoints:
            selected_point_idxs = np.random.choice(8000, self.numpoints, replace=False)
        else:
            selected_point_idxs = np.random.choice(8000, self.numpoints, replace=True)
        selected_data = point[selected_point_idxs]
        selected_label = label[selected_point_idxs]
        return selected_data, selected_label

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    point_data = CarlaDataset(carla_dir='..\\data\\carla', split='train')
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
