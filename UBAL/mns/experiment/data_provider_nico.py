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
    def __init__(self, filename, k):
        self.k = k
        self.data = {1:[], 2:[], 3:[]}
        with open("../nicodata/{}".format(filename)) as f:
            first = f.readline()
            self.map_size = int(first.split(",")[0])
            for line in f:
                split_line = line.split(",")
                label = int(split_line[1])
                values = [float(x) for x in split_line[2:]]
                # print(label)
                # print(values)
                self.data[label].append(np.array(values, dtype=float))
            f.close()
        self.data = apply_kwta(self.data, self.map_size, self.k)


class MSOMDataVis:
    def __init__(self, path, k):
        self.k = k
        self.data = {}
        with open(path) as f:
            first = f.readline()
            self.map_size = int(first.split(",")[0])
            for line in f:
                split_line = line.split(",")
                lgrasp = int(split_line[1])
                lpersp = int(split_line[2])
                values = [float(x) for x in split_line[3:]]
                # print(lgrasp,lpersp)
                # print(len(values))
                if (lgrasp, lpersp) not in self.data:
                    self.data[(lgrasp, lpersp)] = [np.array(values, dtype=float)]
                else:
                    self.data[(lgrasp, lpersp)].append(np.array(values, dtype=float))
            f.close()
        self.data = apply_kwta(self.data, self.map_size, self.k)


if __name__ == "__main__":
    # data = MSOMDataMot("1715358186_pmc.act", 12)
    data = MSOMDataVis("../nicodata/real/{}".format("1715532599_stsp.act"), 12)
    # data = MSOMDataVis("../data/msomnew/{}".format("1513111457_stsp.act"), 12)