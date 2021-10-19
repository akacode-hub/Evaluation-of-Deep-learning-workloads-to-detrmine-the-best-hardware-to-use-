import numpy as np
import scipy.stats
import random
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def calc_pxl(data, mean, cov):

    return scipy.stats.multivariate_normal.pdf(data, mean=mean, cov=cov)

def erm(data, means, covs, num_samples):

    print('***** erm *****')
    m0, m1 = means
    C0, C1 = covs

    pts = data[:2,:].T

    px0_0 = scipy.stats.multivariate_normal.pdf(pts, mean=m0[0,:], cov=C0[0,:,:])
    px0_1 = scipy.stats.multivariate_normal.pdf(pts, mean=m0[1,:], cov=C0[1,:,:])

    px0 = w1*px0_0 + w2*px0_1
    px1 = scipy.stats.multivariate_normal.pdf(pts, mean=m1, cov=C1)

    score = np.divide(px1, px0)
    log_score = np.log(score)
    
def plot_data(data):

    print('***** plot *****')
    l0_ids = np.where(data[2,:]==0)[0]
    l1_ids = np.where(data[2,:]==1)[0]

    data0 = data[:,l0_ids]
    data1 = data[:,l1_ids]

    plt.scatter(data0[0, :], data0[1, :], s=5, color = 'red', label = 'class 0',marker='*')
    plt.scatter(data1[0, :], data1[1, :], s=5, color = 'blue', label = 'class 1', marker='*')

    plt.title("True Label distribution")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

def generate_data_pxgl(prior, means, covs, num_samples):

    m0, m1 = means
    C0, C1 = covs
    
    N0 = int(prior[0]*num_samples)
    N1 = num_samples - N0
    print('N0, N1 ',N0, N1)

    # generate L0
    N00 = int(w1*N0)
    N01 = N0 - N00
    wt_dist = [0]*N00 + [1]*N01
    for i in range(10):
        random.shuffle(wt_dist)
    wt_dist = np.array(wt_dist)

    dist0 = np.random.multivariate_normal(m0[0,:], C0[0,:,:], N0).T
    dist1 = np.random.multivariate_normal(m0[1,:], C0[1,:,:], N0).T
    pxgl0 = np.multiply(1-wt_dist, dist0) + np.multiply(wt_dist, dist1)

    # generate L1
    pxgl1 = np.random.multivariate_normal(m1, C1, N1).T

    ## combine data and label
    labels = [0]*N0 + [1]*N1
    labels = np.reshape(labels, (1, -1))

    pxgl = np.concatenate((pxgl0, pxgl1), axis=1)
    data = np.concatenate((pxgl, labels), axis=0)
    
    return data

if __name__ == "__main__":

    dim = 2

    #priors
    pL = [0.6, 0.4]
    
    #means
    m0 = np.array([[5, 0], [0, 4]]) 
    m1 = [3, 2]

    #covariance
    C0 = np.zeros((2,2,2), dtype=int)
    C0[0,:,:] = np.array([[4, 0], [0, 2]])
    C0[1,:,:] = np.array([[1, 0], [0, 3]])
    C1 = np.array([[2, 0], [0, 2]])

    ## gaus weight
    w1 = 0.5; w2 = 0.5

    # data
    samples_type = {
        'D100': [1e+2],  ## num_samples, data_wt_labels
        'D1k': [1e+3],
        'D10k': [1e+4],
        'D20k': [2e+4],
    }    

    for i, key in enumerate(samples_type.keys()):

        sample_type = samples_type[key]
        num_samples = int(sample_type[0])
        num_samples = 10000

        data_wt_labels = generate_data_pxgl(pL, [m0, m1], [C0, C1], num_samples)
        print(data_wt_labels)
        sample_type.append(data_wt_labels)  
        plot_data(data_wt_labels)

        erm(data_wt_labels, [m0, m1], [C0, C1], num_samples)

        if i==0:break
    


###
# labels = [0]*N0 + [1]*N1
# for i in range(10):
#     random.shuffle(labels)
# labels = np.array(labels)
# print('labels ',labels)