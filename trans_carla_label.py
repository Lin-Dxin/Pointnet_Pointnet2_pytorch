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
    trans_label = [0, 1, 2, 3, 4]

    # 创建生成数据文件目录
    gen_dir = Path('data/carla_t')
    gen_dir.mkdir(exist_ok=True)
    source_folder = os.getcwd() + '\\data\\carla\\'
    target_folder = os.getcwd() + '\\data\\carla_t\\'

    for file in file_list:
        # shutil.copyfile(source_folder + file, target_folder + file)
        raw_data = np.load(source_folder + file)
        # renewdata = [list(raw) for raw in raw_data]
        renewdata = []
        for raw in raw_data:
            if raw[5] in valid_label:
                raw[5] = trans_label[valid_label.index(raw[5])]
                raw = [raw[0], raw[1], raw[2],raw[3],raw[4],raw[5]]
                renewdata.append(raw)
        renewdata = np.asarray(renewdata)
        np.save(target_folder + file, renewdata)
        # data = np.load(target_folder + file)

        print("%s convert complete" % file)
