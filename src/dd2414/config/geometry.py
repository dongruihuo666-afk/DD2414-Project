"""Original Simple-BEV grid constants, in the reference-camera frame."""
import numpy as np
import torch
scene_centroid_x, scene_centroid_y, scene_centroid_z = 0.0, 1.0, 0.0
scene_centroid_py = np.array([scene_centroid_x, scene_centroid_y, scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()
XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
Z, Y, X = 200, 8, 200
