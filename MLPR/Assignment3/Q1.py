import numpy as np
import scipy.stats
import random
import matplotlib.pyplot as plt
import sys
np.set_printoptions(suppress=True)

def set_mean_cov():

    mval = 10
    m0 = np.array([1, 0, 0])*mval
    m1 = np.array([1, 1, 0])*mval
    m2 = np.array([1, 0, 1])*mval
    m3 = np.array([1, 1, 1])*mval

    cval = 10
    C0 = np.eye(dim, dtype=float)*cval
    C1 = np.eye(dim, dtype=float)*cval
    C2 = np.eye(dim, dtype=float)*cval
    C3 = np.eye(dim, dtype=float)*cval

    return [m0, m1, m2, m3], [C0, C1, C2, C3]

def gen_class_samples(num_samples):

    class_dist = np.random.randint(num_labels, size=num_samples)
    class_samples = [np.sum(class_dist==label).astype('int') for label in labels]

    return class_samples

def generate_data_pxgl(priors, means, covs, num_samples):

    m0, m1 = means
    C0, C1 = covs
    
    class_samples = gen_class_samples(num_samples)
    print('class_samples: ',class_samples)

    # generate class data
    for label in clas
    pxgl1 = np.random.multivariate_normal(m1, C1, N1).T
    
    return data, N0, N1

def generate_data_pxgl_samples(samples_type):

    for i, key in enumerate(samples_type.keys()):

        sample_type = samples_type[key]
        num_samples = int(sample_type[0][0])

        data_wt_labels, N0, N1 = generate_data_pxgl(pL, [m0, m1], [C0, C1], num_samples)

        sample_type[1] = [N0, N1]
        sample_type[2] = data_wt_labels

        label_names = ["True label distribution for " + str(num_samples) + " for two classes", "x1", "x2"]
        plot_dist(data_wt_labels, label_names)

    return samples_type

if __name__ == "__main__":

    dim = 3
    labels = [0, 1, 2, 3]
    num_labels = len(labels)
    priors = [0.25, 0.25, 0.25, 0.25]

    samples_type = {
        'D100': [[100]],  
        'D200': [[200]],
        'D500': [[500]],
        'D1000': [[1000]],
        'D2000': [[2000]],
        'D5000': [[5000]],
    }

    means, covs = set_mean_cov()
