import numpy as np
import matplotlib.pyplot as plt

def plot_numperc_data():


    losses = [2.2517, 1.9913, 1.7996, 1.6715, 1.5444, 1.4249, 1.317, 1.2309, 1.147, 1.0883, 1.0333, 0.9813, 0.9659, 0.9299, 0.8882, 0.8843, 0.8375, 0.8432, 0.8, 0.794, 0.781, 0.7618, 0.7312, 0.7272, 0.7144, 0.7239, 0.6947, 0.688, 0.6604, 0.6661, 0.6232, 0.5976, 0.5867, 0.5731, 0.5854, 0.566, 0.5678, 0.5623, 0.5666, 0.5601, 0.5478, 0.5413, 0.5369, 0.5372, 0.5359, 0.5348, 0.5307, 0.5204, 0.5346, 0.5295, 0.5166, 0.5062, 0.5165, 0.5154, 0.5005, 0.5099, 0.5099, 0.514, 0.5028, 0.5042, 0.5085, 0.5075, 0.4925, 0.5001, 0.4963, 0.4948, 0.4975, 0.4941, 0.4921, 0.4853, 0.4661, 0.4761, 0.4751, 0.4762, 0.4718, 0.4798, 0.48, 0.4811, 0.4751, 0.4701, 0.4673, 0.4625, 0.468, 0.479, 0.4723, 0.4713, 0.468, 0.458, 0.4553, 0.4636, 0.4585, 0.456, 0.4485, 0.458, 0.4492, 0.4368, 0.4522, 0.4581, 0.4631, 0.4475]

    num_perc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])*2
    desired_num_perc = num_perc[np.argmin(errs)]
    print('desired_num_perc ', desired_num_perc)
    plt.ylim([0, 1])
    plt.plot(desired_num_perc, accs[np.argmin(errs)], 'rx')
    plt.scatter(num_perc, accs, s=20, color = 'blue')
    plt.title('MLP K-Fold Hyperparameter Validation Performance')
    plt.xlabel('Number of perceptrons in the hidden layer')
    plt.ylabel('MLP accuracy')
    plt.show()

plot_numperc_data()