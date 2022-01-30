import numpy as np
import matplotlib.pyplot as plt

def plot_numperc_data():

    accs = [0.65, 0.775, 0.82, 0.795, 0.825, 0.82, 0.80, 0.826, 0.824, 0.81, 0.824]
    errs = [ 0.59295471,  0.44231551,  0.39638435,  0.42003531,  0.3816584,   0.40209259, 0.41593478,  0.38058893,  0.38119623,  0.40798661,  0.39067709]
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