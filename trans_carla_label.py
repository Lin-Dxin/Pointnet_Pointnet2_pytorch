import os
from pathlib import Path
import shutil
import numpy as np

if __name__ == '__main__':
    # 读取原始数据文件名列表
    carla_dir = 'data/carla'
    file_list = os.listdir(carla_dir)

    # 定义1类、2类、3类label
    valid_label = [1, 7, 8, 10, 11]
    trans_label = [1, 2, 3, 4, 5]

    # 创建生成数据文件目录
    gen_dir = Path('data/carla_t')
    gen_dir.mkdir(exist_ok=True)
    source_folder = os.getcwd() + '\\data\\carla\\'
    target_folder = os.getcwd() + '\\data\\carla_t\\'
    # print(folder)

    for file in file_list:
        # shutil.copyfile(source_folder + file, target_folder + file)
        raw_data = np.load(source_folder + file)
        for data in raw_data:
            if data[5] in valid_label:
                # print(data[5])
                data[5] = trans_label[valid_label.index(data[5])]
                # print(data[5])
            else:
                data[5] = 0
            # print(data[5])
        np.save(target_folder + file, raw_data)
        # data = np.load(target_folder + file)
        print("%s convert complete" % file)
