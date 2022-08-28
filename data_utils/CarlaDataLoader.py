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
                 label_weights=np.random.uniform(size=24)):
        self.split = split
        self.proportion = proportion
        self.carla_dir = carla_dir
        self.transform = transform
        self.label_weights = label_weights

        all_file = os.listdir(self.carla_dir)
        datanum = len(all_file)
        offset = int(datanum * proportion)
        if split == 'train':
            all_file = all_file[:offset]
        if split == 'test':
            all_file = all_file[offset:]
        data = []
        label = []
        for file in all_file:
            plydata = PlyData.read(os.path.join(self.carla_dir, file))
            raw_data = plydata.elements[0].data
            for point in raw_data:
                temp = [point[0], point[1], point[2], point[3]]
                data.append(temp)
                # print(data)
                label.append(point[5])
        self.data_len = len(label)
        self.point_label = torch.LongTensor(label)
        self.point_data = torch.LongTensor(data)
        # labelweights = np.zeros(23)
        # labelweights = labelweights.astype(np.float32)
        # labelweights = labelweights / np.sum(labelweights)
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, item):
        return self.point_data, self.point_label

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    train_dataset = CarlaDataset(carla_dir='..\\data\\carla',split='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True,
                              worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    for data, label in train_dataset:
        print(type(label))
        break


    for point, target in enumerate(train_loader):
        print(point)
        print(target)

        break

