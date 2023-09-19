from os.path import join
import numpy as np
from group_permutation import ClusterPerm


data_shape = [4, 4, 4]
data = np.random.random_sample(data_shape)

chance_maps = {i: [] for i in range(2)}
data = []
c_maps = []
for i in range(2):
    data.append(np.random.random_sample([1, 4 * 4 * 4]))
    for j in range(2):
        d = np.random.random_sample([1, 4 * 4 * 4])
        c_maps.append(d)
        chance_maps[i].append(d)

data = np.mean(data, axis=0).reshape(data_shape)
a = ClusterPerm(data, chance_maps, brain_shape=data_shape)
print(a())
