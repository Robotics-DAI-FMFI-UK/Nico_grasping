import numpy as np
from sklearn.model_selection import KFold

from mns.experiment.data_provider import MSOMDataMot, MSOMDataVis
from mns.experiment.ubal_mns import UBALSim

data_mot = MSOMDataMot(map_size=8, k=10)
data_vis = MSOMDataVis(map_size=12, k=10)
# valuesMot2 = np.concatenate((valuesMot, valuesMot, valuesMot, valuesMot), axis=1)
# valuesVis2 = np.concatenate((valuesVis, dataVis.valuesPerspIndices(1, indices), dataVis.valuesPerspIndices(2, indices), dataVis.valuesPerspIndices(3, indices)), axis=1)

data_mot_list = []
data_vis_list = []
data_labels = []
for grasp in range(3):
    for persp in range(4):
        for i in range(len(data_mot.data[grasp])):
            data_mot_list.append(data_mot.data[grasp][i])
            data_vis_list.append(data_vis.data[(grasp, persp)][i])
            data_labels.append([grasp,persp])
print(data_mot_list)
print(data_labels)
print(len(data_mot_list))

hidden_size = 50
learning_rate = 0.05
init_w_mean = 0.0
init_w_var = 0.5
betas = [1.0, 1.0, 0.9]
gammasF = [float("nan"), 1.0, 1.0]
gammasB = [0.9, 1.0, float("nan")]
max_epoch = 80

experiment = UBALSim(hidden_size, betas, gammasF, gammasB, learning_rate,
                         init_w_mean, init_w_var, max_epoch)
experiment.train_test_one_net(data_mot_list, data_vis_list, data_mot_list, data_vis_list)