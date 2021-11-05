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
    class_samples = [np.sum(class_dist==label_id).astype('int') for label_id in label_ids]

    return class_samples

def generate_data_pxgl(priors, means, covs, num_samples):
    
    class_samples = gen_class_samples(num_samples)
    print('class_samples: ',class_samples)

    # generate class data
    pxgls = np.array([], dtype=np.float).reshape(3,0)
    labels = []
    for label_id in label_ids:
        num_cls_samples = class_samples[label_id]
        mean = means[label_id]
        cov = covs[label_id]
        pxgl = np.random.multivariate_normal(mean, cov, num_cls_samples).T
        pxgls = np.concatenate((pxgls, pxgl), axis=1)
        class_label = [label_id]*num_cls_samples
        labels += class_label
    
    labels = np.array(labels).reshape((1, -1))
    data = np.concatenate((pxgls, labels), axis=0)

    return data, class_samples

def generate_data_pxgl_samples(samples_type):

    for i, key in enumerate(samples_type.keys()):

        sample_type = samples_type[key]
        num_samples = int(sample_type[0][0])

        data_wt_labels, cls_samples = generate_data_pxgl(priors, means, covs, num_samples)
        sample_type[1] = cls_samples
        sample_type[2] = data_wt_labels

        label_names = ["True label distribution for " + str(num_samples) + " for two classes", "x", "y", "z"]
        plot_dist(data_wt_labels, label_names)

    return samples_type

def split_data(data_wt_labels):

    samples = []

    for label_id in label_ids:
        class_ids = np.where(data_wt_labels[-1,:]==label_id)[0]
        cls_samples = data_wt_labels[:,class_ids]    
        samples.append(cls_samples)

    return samples

def plot_dist(data, label_names):

    tname, xname, yname, zname = label_names

    print('***** plot *****')
    samples = split_data(data)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = ['red', 'blue', 'green', 'brown']
    for label_id, sample in enumerate(samples):
        ax.scatter(sample[0, :], sample[1, :], sample[2, :], s=5, color = colors[label_id], label = 'class ' + str(label_id), marker='*')
    
    ax.set_title(tname)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    dim = 3
    label_ids = [0, 1, 2, 3]
    num_labels = len(label_ids)
    priors = [0.25, 0.25, 0.25, 0.25]

    samples_type = {
        'D100': [[100], [], []],  
        'D200': [[200], [], []],
        'D500': [[500], [], []],
        'D1k': [[1000], [], []],
        'D2k': [[2000], [], []],
        'D5k': [[5000], [], []],
        'D100k': [[100000], [], []],
    }

    means, covs = set_mean_cov()

    generate_data_pxgl_samples(samples_type)