import numpy as np

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
    result = np.zeros((map_size, map_size))
    result[win_coords] = 1.0
    return result.reshape(map_size * map_size)


def apply_kwta(data, map_size, k):
    for d in data:
        for index,item in enumerate(data[d]):
            data[d][index] = kwta(item, map_size, k)
    return data


class MSOMDataMot:
    def __init__(self, map_size, k):
        self.map_size = map_size
        self.k = k
        self.data = {0:[], 1:[], 2:[]}
        with open("../data/msom/motor_{}_{}.txt".format(map_size, map_size)) as f:
            prevline_processed = np.arange(map_size * map_size)
            for line in f:
                if "&" in line:
                    label = int(line.split()[1])
                    self.data[label].append(prevline_processed)
                else:
                    prevline_processed = np.array(line.split(), dtype=float)
            f.close()
        self.data = apply_kwta(self.data, self.map_size, self.k)


class MSOMDataVis:
    def __init__(self, map_size, k):
        self.map_size = 12
        self.k = k
        self.data = {}
        with open("../data/msom/visual_{}_{}.txt".format(map_size, map_size)) as f:
            prevline_processed = np.arange(map_size * map_size)
            for line in f:
                if "&" in line:
                    labels = line.split()
                    lgrasp = int(labels[1])
                    lpersp = int(labels[2])
                    if (lgrasp, lpersp) not in self.data:
                        self.data[(lgrasp, lpersp)] = [prevline_processed]
                    else:
                        self.data[(lgrasp, lpersp)].append(prevline_processed)
                else:
                    prevline_processed = np.array(line.split(), dtype=float)
            f.close()
        self.data = apply_kwta(self.data, self.map_size, self.k)


