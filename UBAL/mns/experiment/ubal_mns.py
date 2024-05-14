import time
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

from mns.experiment.util import humanreadible_runtime
from model.UBAL_numpy import UBAL, Sigmoid
from sklearn.model_selection import KFold

class UBALSim:
    def __init__(self, hidden_neurons, betas, gammasF, gammasB, learning_rate,
                 init_w_mean, init_w_variance, max_epoch):
        self.hidden_neurons = hidden_neurons
        self.betas = betas
        self.gammasF = gammasF
        self.gammasB = gammasB
        self.learning_rate = learning_rate
        self.init_w_mean = init_w_mean
        self.init_w_variance = init_w_variance
        self.max_epoch = max_epoch


    def train_test_one_net(self, train_data_x, train_data_y, test_data_x, test_data_y):
        train_data_x = np.stack(train_data_x, axis=-1)
        train_data_y = np.stack(train_data_y, axis=-1)
        test_data_x = np.stack(test_data_x, axis=-1)
        test_data_y = np.stack(test_data_y, axis=-1)
        start_time = time.time()
        sigmoid = Sigmoid()
        act_fun_F = [sigmoid, sigmoid, sigmoid]
        act_fun_B = [sigmoid, sigmoid, sigmoid]
        network = UBAL([len(train_data_x), self.hidden_neurons, len(train_data_y)], act_fun_F, act_fun_B,
                       self.learning_rate, self.init_w_mean, self.init_w_variance,
                       self.betas, self.gammasF, self.gammasB)
        # performance = {"acc_f_train": [], "acc_f_test": [], "acc_b_train": [], "acc_b_test": [],
        #                "mse_f_train": [], "mse_b_train": [], "f1_f_test": [], "f1_b_test": []}
        performance = {"acc_f_test": [], "acc_b_test": [], "mse_f_train": [], "mse_b_train": [],
                       "f1_f_test": [], "f1_b_test": []}
        for epoch in range(self.max_epoch):
            runtime = {"start": time.time(), "end": time.time()}
            indexer = np.random.permutation(len(train_data_x[0]))
            epoch_mse_f = 0
            epoch_mse_b = 0
            for i in indexer:
                input_x = train_data_x[:, indexer[i:i + 1]]
                input_y = train_data_y[:, indexer[i:i + 1]]
                act_FP, act_FE, act_BP, act_BE = network.activation(input_x, input_y)
                network.learning(act_FP, act_FE, act_BP, act_BE)
                epoch_mse_f += mean_squared_error(act_FP[network.d-1], input_y)
                epoch_mse_b += mean_squared_error(act_BP[0], input_x)

            act_FP, act_FE, act_BP, act_BE = network.activation(test_data_x, test_data_y)
            arg_out = (act_FP[network.d-1] > 0.5).astype(np.float_)
            test_acc_f = np.sum(arg_out == test_data_y) / (len(test_data_y[0]) * len(test_data_y))
            performance["acc_f_test"].append(test_acc_f)
            test_f1_f = f1_score(test_data_y, arg_out, average=None).mean()
            performance["f1_f_test"].append(test_f1_f)
            arg_out = (act_BP[0] > 0.5).astype(np.float_)
            test_score_b = np.sum(arg_out == test_data_x) / (len(test_data_x[0]) * len(test_data_x))
            performance["acc_b_test"].append(test_score_b)
            test_f1_b = f1_score(test_data_x, arg_out, average=None).mean()
            performance["f1_b_test"].append(test_f1_b)

            epoch_mse_f /= len(train_data_x)
            epoch_mse_b /= len(train_data_x)
            performance["mse_f_train"].append(epoch_mse_f)
            performance["mse_b_train"].append(epoch_mse_b)

            epoch += 1
            end_time = time.time()
            # if epoch % 100 == 0:
            print("Epc {}\tAcc_F: {:.3f}%. F1_F: {:.3f}%. Acc_B: {:.3f}%. F1_B: {:.3f}%. MSE_train_F: {:.4f}. "
                  "MSE_train_B: {:.4f}.  Time: {:.1f} seconds".format(
                epoch + 1, test_acc_f * 100., test_f1_f * 100., test_score_b * 100., test_f1_b * 100.,
                epoch_mse_f, epoch_mse_b, end_time - start_time))
            # print(".")
            start_time = end_time
            runtime["end"] = end_time
        runtime_total = runtime["end"] - runtime["start"]
        print("Total runtime: {}".format(humanreadible_runtime(runtime_total)))
        return performance

    def train_test_crossval(self, folds, reps, data_labels, data_mot_list, data_vis_list):
        performance = []
        kfold = KFold(n_splits=folds, shuffle=True)
        for f, (train_index, test_index) in enumerate(kfold.split(data_labels)):
            print("Fold", f + 1, "of", folds)
            data_train_mot = [data_mot_list[i] for i in train_index]
            data_train_vis = [data_vis_list[i] for i in train_index]
            data_test_mot = [data_mot_list[i] for i in test_index]
            data_test_vis = [data_vis_list[i] for i in test_index]
            for r in range(reps):
                performance.append(self.train_test_one_net(data_train_mot, data_train_vis, data_test_mot, data_test_vis))
        return performance

