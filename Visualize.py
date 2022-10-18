import os
import open3d as o3d
import numpy as np


def transform(files, source_dir, target_dir):
    for f in files:
        frame = f.split('.')[0]
        data = np.load(source_dir + f)
        renew_data = [list(raw) for raw in data]
        renew_data = np.asarray(renew_data)  # 可以正常索引的数据
        # print(renew_data[:][:3])
        # print(renew_data.shape)

        txt_file_name = frame + ".txt"
        np.savetxt(target_dir + txt_file_name, renew_data)



def visualize_pc(files):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    colors_0 = np.random.randint(255, size=(23, 3)) / 255.
    pcd = o3d.geometry.PointCloud()
    to_reset = True
    vis.add_geometry(pcd)
    for f in files:
        # frame = f.split('.')[0]
        # txt_file_name = frame + ".txt"
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters('viewpoint.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        # pcd = o3d.io.read_point_cloud("./data/carla_txt/" + f, format='xyz')
        points = np.load("./data/carla_t/" + f)
        renew_data = [list(raw) for raw in points]
        points = np.asarray(renew_data)

        # pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # 为各个真实标签指定颜色
        colors = colors_0[points[:, -1].astype(np.uint8)]
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # 点云显示
        # o3d.visualization.draw_geometries(pcd)  # 窗口高度
        # pcd = np.asarray(pcd.points)
        # print(pcd)
        # poincloud.points = o3d.utility.Vector3dVector(pcd)
        vis.update_geometry(pcd)
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
        vis.poll_events()
        vis.update_renderer()


if __name__ == '__main__':
    SourceDir = "./data/carla_t/"
    TargetDir = "./data/carla_t/"
    Files = os.listdir(SourceDir)
    TFiles = os.listdir(TargetDir)
    # pcd = o3d.io.read_point_cloud(TargetDir+TFiles[400], format='xyz')  # 传入自己当前的pcd文件
    # save_view_point(pcd, "viewpoint.json")  # 保存好得json文件位置
    # load_view_point(pcd, "viewpoint.json")  # 加载修改时较后的pcd文件
    # transform(Files, SourceDir, TargetDir)
    visualize_pc(TFiles)

