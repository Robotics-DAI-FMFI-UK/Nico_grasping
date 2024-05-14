import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from mns.experiment.data_provider_nico import MSOMDataMot, MSOMDataVis
from mns.experiment.ubal_mns import UBALSim

k_mot = 8
k_vis = 8
data_mot = MSOMDataMot("../nicodata/real/{}".format("1715532599_pmc.act"), k_mot)
data_vis = MSOMDataVis("../nicodata/real/{}".format("1715532599_stsp.act"), k_vis)

data_mot_list = []
data_vis_list = []
data_labels = []
for i in data_mot.data:
    for j in range(len(data_mot.data[i])):
        data_mot_list.append(data_mot.data[i][j])
        data_vis_list.append(data_vis.data[(i,0)][j])
        data_labels.append(i)
# print(data_mot_list)
# print(data_vis_list)
# print(data_labels)
# exit()

hidden_size = 70
learning_rate = 0.05
init_w_mean = 0.0
init_w_var = 0.5
betas = [1.0, 1.0, 0.9]
gammasF = [float("nan"), 1.0, 1.0]
gammasB = [0.9, 1.0, float("nan")]
max_epoch = 25
folds = 5
repetitions = 10

experiment = UBALSim(hidden_size, betas, gammasF, gammasB, learning_rate,
                         init_w_mean, init_w_var, max_epoch)
performance = experiment.train_test_crossval(folds, repetitions, data_labels, data_mot_list, data_vis_list)
# {"acc_f_test": [], "acc_b_test": [], "mse_f_train": [], "mse_b_train": [], "f1_f_test": [], "f1_b_test": []}

# print(performance)

performance_avg = {"acc_f_test": np.zeros(shape=(folds*repetitions,max_epoch)),
                   "acc_b_test": np.zeros(shape=(folds*repetitions,max_epoch)),
                   "mse_f_train": np.zeros(shape=(folds*repetitions,max_epoch)),
                   "mse_b_train": np.zeros(shape=(folds*repetitions,max_epoch))}

for index,res in enumerate(performance):
    for key in performance_avg:
        performance_avg[key][index] = np.array(res[key])

# print(performance_avg)
# print(performance_avg["acc_f_test"].shape)
# exit()

for key in performance_avg:
    k = 0
    if "f" in key:
        k = k_vis
    else:
        k = k_mot
    with open("../output/{}_k{}.txt".format(key, k), 'w+') as file:
        print("writing",key)
        perf_mean = np.mean(performance_avg[key], axis=0)
        perf_std = np.std(performance_avg[key], axis=0)
        file.write("x y err\n")
        for i in range(max_epoch):
            file.write("{} {} {}\n".format(i,perf_mean[i], perf_std[i]))
        print(key,perf_mean,perf_std)

# print("acc_b_test")
# acc_b_mean = np.mean(performance_avg["acc_b_test"], axis=0)
# acc_b_std = np.std(performance_avg["acc_b_test"], axis=0)
# for i in range(max_epoch):
#     print(i, acc_b_mean[i], acc_f_std[i])
#
# print("mse_f_train")
# acc_f_mean = np.mean(performance_avg["mse_f_train"], axis=0)
# acc_f_std = np.std(performance_avg["mse_f_train"], axis=0)
# for i in range(max_epoch):
#     print(i, acc_f_mean[i], acc_f_std[i])
#
# print("mse_b_train")
# acc_b_mean = np.mean(performance_avg["mse_b_train"], axis=0)
# acc_b_std = np.std(performance_avg["mse_b_train"], axis=0)
# for i in range(max_epoch):
#     print(i, acc_b_mean[i], acc_f_std[i])

# plt.figure()
# plt.plot(list(range(max_epoch)),np.mean(performance_avg["acc_f_test"], axis=0), label="acc_f_test")
# plt.plot(list(range(max_epoch)),np.mean(performance_avg["acc_b_test"], axis=0), label="acc_b_test")
# # plt.plot(list(range(max_epoch)),performance["mse_f_train"], label="mse_f_train")
# # plt.plot(list(range(max_epoch)),performance["mse_b_train"], label="mse_b_train")
# plt.savefig("single_net_training_plot.{}.png".format(int(time.time())))
# plt.show()