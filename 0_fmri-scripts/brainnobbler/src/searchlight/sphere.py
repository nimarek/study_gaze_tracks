"""
Script to plot spheres with the only
purpose to show how they look like 
and the number of voxels per sphere
"""
import matplotlib.pyplot as plt
import numpy as np


def mk_sphere(shape, center, radius):
    """make spheres, just for demonstration"""
    coord = np.indices(shape)
    sphere = (
        np.sqrt(
            (coord[0] - center[0]) ** 2
            + (coord[1] - center[1]) ** 2
            + (coord[2] - center[2]) ** 2
        )
        <= radius
    )
    sphere_idx = np.asarray(np.where(sphere))
    return sphere_idx.T, sphere


radii = [1, 2, 3, 4]
for radius in radii:
    idx, sphere = mk_sphere([20, 20, 20], [10, 10, 10], radius)
    print(idx.T)
    axs = plt.figure().add_subplot(projection="3d")
    axs.voxels(sphere, edgecolor="k")
    plt.title(f"Searchlight of radius {radius} - n.voxels: {len(idx)}")
    plt.show()
