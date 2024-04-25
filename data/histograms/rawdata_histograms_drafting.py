import json
import numpy as np
import matplotlib.pyplot as plt


with open("my_data.rawdata") as f:
    loaded_data = json.load(f)

    # item_count = 0
    # for row in loaded_data:
    #     item_count += 1
    # print(item_count)

    right_pos_list = []
    for row in loaded_data:
        # print(row['right-pos'])
        # print(row['left-pos'])
        # tmp_arr = np.fromstring(row['right-pos'], dtype=float, sep=' ')
        # print(tmp_arr.shape)
        # np.append([right_pos_data], tmp_arr, axis=0)
        right_pos_list.append([float(i) for i in row['right-pos'].split()])
    right_pos_data = np.array(right_pos_list)
    # print(right_pos_data)
    # print(right_pos_data.shape)
    # for joint_no in range(16):
    #     tmp = right_pos_data[:, [0]]
    #     tmp = tmp.transpose().flatten()
    #     print(tmp)
    #     print(tmp.shape)
    #     # hist = np.histogram(tmp, bins='auto')
    #     # print(hist)
    #     _ = plt.hist(tmp, bins='auto')
    #     plt.title("Histogram for joint ",joint_no)
    #     plt.show()
    #     print(np.median(tmp))

    num_row = 4
    num_col = 4
    fig, axes = plt.subplots(num_row, num_col, figsize=(3 * num_col, 2.5 * num_row))
    for joint_no in range(16):
        joint_data = right_pos_data[:, [joint_no]]
        joint_data = joint_data.transpose().flatten()
        ax = axes[joint_no // num_col, joint_no % num_col]
        ax.hist(joint_data, bins='auto')
        ax.set_title("Histogram for joint {}".format(joint_no+1))
    plt.tight_layout()
    # plt.savefig("joints_right.svg", format="svg")
    plt.savefig("joints_right.png", format="png")
    plt.show()
