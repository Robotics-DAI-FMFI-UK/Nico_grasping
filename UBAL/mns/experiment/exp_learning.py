import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from mns.experiment.data_provider import MSOMDataMot, MSOMDataVis
from mns.experiment.ubal_mns import UBALSim

data_mot = MSOMDataMot(map_size=8, k=10)
data_vis = MSOMDataVis(map_size=12, k=10)
# valuesMot2 = np.concatenate((valuesMot, valuesMot, valuesMot, valuesMot), axis=1)
# valuesVis2 = np.concatenate((valuesVis, dataVis.valuesPerspIndices(1, indices), dataVis.valuesPerspIndices(2, indices), dataVis.valuesPerspIndices(3, indices)), axis=1)

input_mot = []
for i in data_mot.data:
    input_mot.extend(data_mot.data[i])
print(len(input_mot))
# print(input_mot)

input_vis = []
for i in data_mot.data:
    input_vis.extend(data_vis.data[(i,0)])
print(len(input_vis))

hidden_size = 50
learning_rate = 0.05
init_w_mean = 0.0
init_w_var = 0.5
betas = [1.0, 1.0, 0.9]
gammasF = [float("nan"), 1.0, 1.0]
gammasB = [0.9, 1.0, float("nan")]
max_epoch = 100

experiment = UBALSim(hidden_size, betas, gammasF, gammasB, learning_rate,
                         init_w_mean, init_w_var, max_epoch)
# performance = experiment.train_test_one_net(input_mot, input_vis, input_mot, input_vis)
performance = experiment.train_test_crossval(input_mot, input_vis, input_mot, input_vis)
# {"acc_f_test": [], "acc_b_test": [], "mse_f_train": [], "mse_b_train": [], "f1_f_test": [], "f1_b_test": []}

print(len(performance["acc_f_test"]))
print(len(performance["acc_b_test"]))

plt.figure()
plt.plot(list(range(max_epoch)),performance["acc_f_test"], label="acc_f_test")
plt.plot(list(range(max_epoch)),performance["acc_b_test"], label="acc_b_test")
plt.plot(list(range(max_epoch)),performance["mse_f_train"], label="mse_f_train")
plt.plot(list(range(max_epoch)),performance["mse_b_train"], label="mse_b_train")
plt.savefig("single_net_training_plot.{}.png".format(int(time.time())))
plt.show()