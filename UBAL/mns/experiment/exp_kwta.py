import numpy as np
from mns.experiment.data_provider import MSOMDataMot, MSOMDataVis
from mns.experiment.ubal_mns import UBALSim

data_mot = MSOMDataMot(map_size=8, k=10)
data_vis = MSOMDataVis(map_size=12, k=10)
# valuesMot2 = np.concatenate((valuesMot, valuesMot, valuesMot, valuesMot), axis=1)
# valuesVis2 = np.concatenate((valuesVis, dataVis.valuesPerspIndices(1, indices), dataVis.valuesPerspIndices(2, indices), dataVis.valuesPerspIndices(3, indices)), axis=1)
folds = 5

data_mot_list = []
data_vis_list = []
data_labels = []
for i in data_mot.data:
    for j in range(len(data_mot.data[i])):
        data_mot_list.append(data_mot.data[i][j])
        data_vis_list.append(data_vis.data[(i,0)][j])
        data_labels.append(i)
print(data_mot_list)
print(data_vis_list)
print(data_labels)

hidden_size = 50
learning_rate = 0.05
init_w_mean = 0.0
init_w_var = 0.5
betas = [1.0, 1.0, 0.9]
gammasF = [float("nan"), 1.0, 1.0]
gammasB = [0.9, 1.0, float("nan")]
max_epoch = 120




