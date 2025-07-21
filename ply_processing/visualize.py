import open3d as o3d
# visualization of point clouds.
pcd = o3d.io.read_point_cloud('/home/jeff/Research/Point_Cloud/Minkowski/Result/conquer_ply/reconstruct/GT/training_data/longdress_vox10_rec_1060.ply')
o3d.visualization.draw_geometries([pcd])
