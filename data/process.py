# import numpy as np
# points = np.loadtxt("exported_cloud.txt", usecols=(0, 1))  # (N, 2)
# np.save("pointcloud_2d.npy", points)

import numpy as np, json
pcd = np.load("pointcloud_2d.npy")
with open("/home/yan/myProjects/RLHW/RL_assignment/data/eval_initials_100.json") as f:
    initials = json.load(f)
print(f"Point cloud: {pcd.shape}, Initials: {len(initials)}")