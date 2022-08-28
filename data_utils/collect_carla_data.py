import os
from plyfile import PlyData
import numpy as np

if __name__ == '__main__':
    carla_dir = '..\\data\\carla'
    all_file = os.listdir(carla_dir)
    data = []  # x, y, z, v

    label = []  # label
    cnt = 0
    for file in all_file:
        plydata = PlyData.read(os.path.join(carla_dir, file))
        raw_data = plydata.elements[0].data
        for point in raw_data:
            temp = [point[0], point[1], point[2], point[3]]
            data.append(temp)

            label.append(point[5])

    for data_ in data[:9]:
        print(data_)
    print(label[:9])
