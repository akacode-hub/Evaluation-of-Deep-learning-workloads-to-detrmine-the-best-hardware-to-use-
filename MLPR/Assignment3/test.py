
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt

def wilcoxon():

    d1 = [0.28, 0.34, 0.22, 0.2, 0.18, 0.26, 0.2, 0.12, 0.26, 0.28]
    d2 = [0.23, 0.25, 0.28, 0.22, 0.2, 0.18, 0.24, 0.13, 0.22, 0.22]

    d1 = np.array(d1)
    d2 = np.array(d2)
    diff = d1-d2
    print('diff ',diff)
    w, p = wilcoxon(diff)
    print(w, p)

def plot_pe_perc_data():

    poe100 = [0.63, 0.54, 0.51, 0.33, 0.3,  0.33, 0.27, 0.24]
    poe200 = [0.61,  0.57,  0.44,  0.39,  0.31,  0.275, 0.24,  0.235]
    poe500 = [0.582, 0.55,  0.426, 0.31,  0.232, 0.188, 0.184, 0.17 ]
    poe1k = [0.621, 0.482, 0.411, 0.293, 0.228, 0.211, 0.218, 0.2  ]
    poe2k = [0.5625, 0.5045, 0.446,  0.3275, 0.264,  0.2425, 0.2185, 0.2125]
    poe5k = [0.5732, 0.5612, 0.425,  0.3314, 0.2556, 0.2154, 0.2028, 0.197 ]    
    percs = [1, 2, 4, 8, 16, 25, 35, 50]
    num_samples = [100, 200, 500, 1000, 2000, 5000]
    poes = [poe100, poe200, poe500, poe1k, poe2k, poe5k]

    for i in range(len(num_samples)):
        poe = poes[i]
        num_sample = num_samples[i]
        plt.scatter(percs, poe, s=20, color = 'blue')
        plt.title('Probability of Error vs number of perceptrons for ' + str(num_sample) + ' samples')
        plt.xlabel('Number of perceptrons in first layer')
        plt.ylabel('Probability of Error')
        plt.show()

def plot_pe_data(poes, num_samples):

    
    plt.scatter(num_samples, poes, s=20, color = 'blue')
    plt.axhline(y=0.1372, color='r', linestyle='-')
    plt.title('Probability of Error on test data vs number of training samples')
    plt.xlabel('Number of training samples')
    plt.ylabel('Probability of Error')
    plt.legend(["Optimal pFE", "MLP pFE"])
    plt.show()

def plot_hist():

    #num_gmm_freq = [4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4]
    #num_samples = 10000
    # num_gmm_freq = [4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 6, 5, 5, 4, 6, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 5, 4, 4, 5, 5, 4, 5, 4, 4, 5]
    # num_samples = 1000

    # num_gmm_freq = [4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4]
    # num_samples = 100

    num_gmm_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    num_samples = 10
    
    print('num_gmm_freq: ',num_gmm_freq)
    n_bins = 6
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlim([1, 6])
    ax.hist(num_gmm_freq, bins=n_bins)
    plt.title('Frequency of model order across 35 experiments for ' + str(num_samples) + ' samples')
    plt.xlabel('GMM model orders')
    plt.ylabel('Frequency of GMM model order')
    plt.show()
    # plt.savefig(str(num_samples) + '_' + str(num_time) + '.png')

if __name__ == "__main__":              

    #50
    poe_test = [0.20, 0.156, 0.152, 0.149, 0.147, 0.145]
    num_samples = [100, 200, 500, 1000, 2000, 5000]

    #plot_pe_data(poe_test, num_samples)
    plot_hist()