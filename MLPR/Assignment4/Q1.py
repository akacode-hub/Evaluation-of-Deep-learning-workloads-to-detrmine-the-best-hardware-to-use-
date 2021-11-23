import numpy as np
import cv2
import matplotlib.pyplot as plt

def gen_data(num_samples, priors):

    noise = np.random.multivariate_normal(noise_mu, noise_sigma, num_samples).T
    
    thetas = np.random.uniform(-np.pi, np.pi, num_samples)
    sample_data = np.random.uniform(0.0, 1.0, num_samples)
    num_cls1 = np.sum((sample_data > priors[0]).astype('int'))
    num_cls2 = num_samples - num_cls1

    print('num_cls1: ',num_cls1)
    print('num_cls2: ',num_cls2)

    cos_thetas = np.cos(thetas).reshape((-1, 1))
    sine_thetas = np.sin(thetas).reshape((-1, 1))

    vecs = np.hstack((cos_thetas, sine_thetas)).T

    data_cls1 = r[0] * vecs[:, :num_cls1] + noise[:, :num_cls1]
    data_cls2 = r[1] * vecs[:, num_cls1:] + noise[:, num_cls1:]

    label_cls1 = np.zeros((1, num_cls1), dtype='int')
    label_cls2 = np.ones((1, num_cls2), dtype='int')
    
    data_wt_cls1 = np.vstack((data_cls1, label_cls1))
    data_wt_cls2 = np.vstack((data_cls2, label_cls2))

    return np.hstack((data_wt_cls1, data_wt_cls2))

def split_data(data_wt_labels, label_ids):

    samples = []

    for label_id in label_ids:
        class_ids = np.where(data_wt_labels[-1,:]==label_id)[0]
        cls_samples = data_wt_labels[:,class_ids]    
        samples.append(cls_samples)

    return samples

def plot_data(data_wt_labels, label_ids):

    fig = plt.figure()
    ax = fig.add_subplot()

    samples = split_data(data_wt_labels, label_ids)

    colors = ['red', 'blue', 'green', 'brown']
    for label_id, sample in enumerate(samples):
        ax.scatter(sample[0, :], sample[1, :], s=5, color = colors[label_id], label = 'class ' + str(label_id), marker='*')
    
    ax.set_title('Data distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.legend()
    plt.show()

if __name__ == "__main__":

    priors = [0.6, 0.4]
    label_ids = [0, 1]
    num_train_samples = 1000
    num_test_samples = 10000
    noise_mu = [0, 0]
    noise_sigma = np.eye(2, dtype=float)
    r = [2, 4]

    ## train
    train_wt_cls = gen_data(num_train_samples, priors)
    plot_data(train_wt_cls, label_ids)

    ## test
    test_wt_cls = gen_data(num_test_samples, priors)
    plot_data(test_wt_cls, label_ids)