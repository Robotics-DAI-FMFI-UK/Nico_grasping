import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def smallest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def kwta(input, map_size, k):
    # print(len(input))
    input_as_map = input.reshape(map_size, map_size)
    win_coords = smallest_indices(input_as_map, k)
    result = np.zeros((map_size,map_size))
    result[win_coords] = 1.0
    return result.reshape(map_size*map_size)



map_size = 16
k = 16
data = {}
with open("msom/visual_{}_{}.txt".format(map_size,map_size)) as f:
    prevline_processed = np.arange(map_size*map_size)
    for line in f:
        if "&" in line:
            labels = line.split()
            lgrasp = int(labels[1])
            lpersp = int(labels[2])
            if (lgrasp,lpersp) not in data:
                data[(lgrasp,lpersp)] = [prevline_processed]
            else:
                data[(lgrasp,lpersp)].append(prevline_processed)
        else:
            prevline_processed = np.array(line.split(), dtype=float)
    f.close()

print(data.keys())

# map_size = 8
# k = 16
# data = {0:[], 1:[], 2:[]}
# with open("msom/motor_{}_{}.txt".format(map_size,map_size)) as f:
#     prevline_processed = np.arange(map_size*map_size)
#     for line in f:
#         if "&" in line:
#             label = int(line.split()[1])
#             data[label].append(prevline_processed)
#         else:
#             prevline_processed = np.array(line.split(), dtype=float)
#     f.close()
# print(np.shape(data[1][1]))
# print(len(data[0]) + len(data[1]) + len(data[2]))

for d in data:
    for item in data[d]:
        item = kwta(item, map_size, k)
        # print(item)

data_pattern = np.reshape(item, (-1, map_size))
print(data_pattern)

# create discrete colormap
cmap = colors.ListedColormap(['white', 'blue'])
bounds = [0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(data_pattern, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(0.5, data_pattern.shape[1], 1));
ax.set_yticks(np.arange(0.5, data_pattern.shape[0], 1));

plt.show()