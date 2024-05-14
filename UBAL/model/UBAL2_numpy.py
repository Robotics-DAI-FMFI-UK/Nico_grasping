import numpy as np
from scipy.special import expit


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, net):
        # return 1.0 / (1.0 + np.exp(-net))
        return expit(net)


class SoftMax:
    def __init__(self):
        pass

    def __call__(self, net):
        e_net = np.exp(net - np.max(net))
        e_denom0 = e_net.sum(axis=0, keepdims=True)
        result = e_net / e_denom0
        return result


class UBAL2:
    def __init__(self, layers, act_fun_f, act_fun_b, learning_rate, init_w_mean, init_w_variance):
        super(UBAL2, self).__init__()
        self.arch = layers
        self.d = len(self.arch)
        self.act_fun_F = act_fun_f
        self.act_fun_B = act_fun_b
        self.learning_rate = learning_rate
        self.init_weight_mean = init_w_mean
        self.init_weight_variance = init_w_variance
        self.weightsF = []
        for i in range(self.d - 1):
            self.weightsF.append(np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[i+1], self.arch[i]+1)))
        self.weightsB = []
        for i in range(self.d - 1):
            self.weightsB.append(np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[i],self.arch[i+1]+1)))

    def add_bias(self, input_array):
        return np.vstack([input_array, np.ones(len(input_array[0]))])

    def activation(self, input_x, input_y):
        act_fp = [None] * self.d
        act_bp = [None] * self.d
        act_fe = [None] * self.d
        act_be = [None] * self.d

        act_fp[0] = input_x
        for i in range(1, self.d):
            act_fp[i] = self.act_fun_F[i](np.dot(self.weightsF[i-1], self.add_bias(act_fp[i-1])))
            act_fe[i-1] = self.act_fun_B[i](np.dot(self.weightsB[i-1], self.add_bias(act_fp[i])))

        act_bp[self.d-1] = input_y
        for i in range(self.d-1, 0, -1):
            act_bp[i-1] = self.act_fun_F[i](np.dot(self.weightsB[i-1], self.add_bias(act_bp[i])))
            act_be[i] = self.act_fun_B[i](np.dot(self.weightsF[i-1],self.add_bias(act_bp[i-1])))

        return act_fp, act_fe, act_bp, act_be

    def learning(self, act_fp, act_fe, act_bp, act_be):
        for l in range(self.d - 1):
            k = l + 1
            weight_update_F = self.learning_rate * np.dot(self.add_bias(act_fp[l]), (act_bp[k] - act_fp[k]).transpose())
            weight_update_B = self.learning_rate * np.dot(self.add_bias(act_bp[k]), (act_bp[l] - act_fe[l]).transpose())
            self.weightsF[l] += weight_update_F.transpose()
            self.weightsB[l] += weight_update_B.transpose()

    def activation_fp_last(self, input_x):
        act_fp = input_x
        for i in range(1, self.d):
            act_fp = self.act_fun_F[i](np.dot(self.weightsF[i - 1],self.add_bias(act_fp)))
        return act_fp