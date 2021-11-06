import numpy as np
import scipy.stats
import random
import matplotlib.pyplot as plt
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
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

def gen_class_samples(num_samples, label_ids):
    
    ##equal prior
    num_labels = len(label_ids)
    class_dist = np.random.randint(num_labels, size=num_samples)
    class_samples = [np.sum(class_dist==label_id).astype('int') for label_id in label_ids]

    return class_samples

def generate_data_pxgl(priors, means, covs, num_samples, label_ids):

    class_samples = gen_class_samples(num_samples, label_ids)
    print('class_samples: ',class_samples, ' sum ', sum(class_samples))

    # generate class data
    pxgls = np.array([], dtype=float).reshape(3,0)
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

def generate_data_pxgl_samples(samples_type, priors, means, covs, label_ids):

    for i, key in enumerate(samples_type.keys()):

        sample_type = samples_type[key]
        num_samples = int(sample_type[0][0])

        data_wt_labels, cls_samples = generate_data_pxgl(priors, means, covs, num_samples, label_ids)
        sample_type[1] = cls_samples
        sample_type[2] = data_wt_labels

        label_names = ["True label distribution for " + str(num_samples) + " for two classes", "x", "y", "z"]
        plot_dist(data_wt_labels, label_names, label_ids)

    return samples_type

def plot_dist(data, label_names, label_ids):

    tname, xname, yname, zname = label_names

    print('***** plot *****')
    samples = split_data(data, label_ids)

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

def split_data(data_wt_labels, label_ids):

    samples = []

    for label_id in label_ids:
        class_ids = np.where(data_wt_labels[-1,:]==label_id)[0]
        cls_samples = data_wt_labels[:,class_ids]    
        samples.append(cls_samples)

    return samples

def calc_theoretical_classifier(sample_type):

    num_samples = sample_type[0][0]
    cls_samples = sample_type[1]
    data_wt_labels = sample_type[2]

    data = data_wt_labels[:3,:].T #(N, 3)
    labels = data_wt_labels[3,:]

    eval_pxgls = np.zeros((num_labels, num_samples), dtype=float) ##(4, N)
    for label_id in label_ids:
        eval_pxgl = scipy.stats.multivariate_normal.pdf(data, mean=means[label_id], cov=covs[label_id])
        eval_pxgls[label_id] = eval_pxgl

    priors_np = np.array(priors)
    px = np.matmul(priors_np.reshape(1,-1), eval_pxgls)  ##(1, N)
    
    stack_px = np.zeros((num_labels, num_samples), dtype=float)
    for label_id in label_ids:
        stack_px[label_id] = px

    plgx = priors_np.reshape(-1, 1)*eval_pxgls/stack_px ## class posterior(4, N)
    risks = np.matmul(loss_mat, plgx)

    decisions = np.argmin(risks, axis=0) 

    correct_ids, incorrect_ids = [], [] 
    for label_id in label_ids:
        label_pids = (labels == label_id)
        correct_cls_bool = ((label_pids)*(decisions == label_id)).astype('int')
        incorrect_cls_bool = ((label_pids)*(decisions != label_id)).astype('int')
        correct_class_ids = np.where(correct_cls_bool == 1)[0]
        incorrect_class_ids = np.where(incorrect_cls_bool == 1)[0]
        correct_ids.append(correct_class_ids)
        incorrect_ids.append(incorrect_class_ids)

    prob_error = 1.0*np.sum((decisions != labels).astype('int'))/num_samples
    prob_error = np.round(prob_error, 4)
    print('prob_error: ',prob_error)
    
    plot_decision(data_wt_labels, correct_ids, incorrect_ids)

def plot_decision(data, correct_ids, incorrect_ids):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    markers = ['o', 'v', 's', 'P']
    for label_id in label_ids:
        ax.scatter(data[0, correct_ids[label_id]], data[1, correct_ids[label_id]], data[2, correct_ids[label_id]], s=5, color = 'green', label = 'correct class' + str(label_id), marker=markers[label_id])
        ax.scatter(data[0, incorrect_ids[label_id]], data[1, incorrect_ids[label_id]], data[2, incorrect_ids[label_id]], s=5, color = 'red', label = 'incorrect class' + str(label_id), marker=markers[label_id])
    
    ax.set_title('MAP classification Results')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def get_model(first_num_nodes):

    model = Sequential()

    # first layer
    fc1_act = Dense(units = first_num_nodes, kernel_initializer = 'random_uniform', activation = 'elu')
    model.add(fc1_act)

    # Second layer
    fc2_act = Dense(units = num_labels, kernel_initializer = 'random_uniform', activation = 'softmax')
    model.add(fc2_act)

    model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model

def calc_pe(label, prediction):

    num_samples = label.shape[0]
    acc = np.sum((label == prediction).astype('int'))/num_samples
    error = 1 - acc

    return error

def MOS(sample_type, kfold, num_perc_lst):

    num_samples = sample_type[0][0]
    cls_samples = sample_type[1]
    data_wt_labels = sample_type[2]
    data_wt_labels = data_wt_labels[:, np.random.permutation(data_wt_labels.shape[1])] #shuffle

    data = data_wt_labels[:3,:].T #(N, 3)
    labels = data_wt_labels[3,:].T

    data = data.reshape((kfold, -1, 3))
    labels = labels.reshape((kfold, -1))
    
    num_val = num_samples/kfold
    num_train = num_samples - num_val

    perc_lst = []
    for num_perc in num_perc_lst:

        err_lst = []
        for val_idx in range(kfold):

            # train
            train_data = np.concatenate((data[0:val_idx], data[val_idx+1:]), axis=0)
            train_labels = np.concatenate((labels[0:val_idx], labels[val_idx+1:]), axis=0)

            # val
            val_data = data[val_idx].reshape((1, -1, 3))
            val_labels = labels[val_idx]

            #data shape summary
            # print('train data shape ',train_data.shape)
            # print('train label shape ',train_labels.shape)
            # print('val data shape ',val_data.shape)
            # print('val labels shape ',val_labels.shape)

            # get model
            model = get_model(num_perc)

            # train
            model.fit(train_data, train_labels, batch_size = 10, epochs = 100, verbose=0)
            
            # validate
            val_pred = model.predict(val_data)
            val_pred = np.argmax(val_pred, axis=2)
            val_pred = np.squeeze(val_pred, axis=0)

            err = calc_pe(val_labels, val_pred)
            print('num_samples:', num_samples,' num_perc: ',num_perc,' val idx: ', val_idx, ' error: ', np.round(err, 4))
            err_lst.append(err)

        mean_err = np.mean(np.array(err_lst))
        std_err = np.std(np.array(err_lst))
        print('num_samples:', num_samples, ' num_perc: ',num_perc,' mean error: ', np.round(mean_err, 4), ' std error: ',np.round(std_err, 4))
        perc_lst.append(mean_err)

    perc_lst = np.array(perc_lst)
    print('pe for each perceptron: ', perc_lst)
    desired_num_perc = np.argmin(perc_lst)

    return desired_num_perc

def train_kfoldMLP(train_sample_type, val_sample_type, kfold, num_perc_lst):

    num_samples = train_sample_type[0][0]
    cls_samples = train_sample_type[1]
    data_wt_labels = train_sample_type[2]
    
    data_wt_labels = data_wt_labels[:, np.random.permutation(data_wt_labels.shape[1])] #shuffle
    data = data_wt_labels[:3,:].T #(N, 3)
    labels = data_wt_labels[3,:].T

    # Model Order Selection
    desired_num_perc = MOS(train_sample_type, kfold, num_perc_lst)

    # get model
    model = get_model(desired_num_perc)

    # train
    model.fit(data, labels, batch_size = 10, epochs = 100, verbose=0)

    # validate
    val_err = validate(val_sample_type, model)

    print('num_samples: ',num_samples,' desired_num_perc: ',desired_num_perc,' val_err: ', val_err)

def validate(sample_type, model):

    num_samples = sample_type[0][0]
    cls_samples = sample_type[1]
    data_wt_labels = sample_type[2]
    data = data_wt_labels[:3,:].T #(N, 3)
    labels = data_wt_labels[3,:].T

    prediction = model.predict(data, workers=4, use_multiprocessing = True)

    prediction = np.argmax(prediction, axis=1)
    err = calc_pe(labels, prediction)

    return err

if __name__ == "__main__":              

    dim = 3
    label_ids = [0, 1, 2, 3]
    num_labels = len(label_ids)
    priors = [0.25, 0.25, 0.25, 0.25]
    loss_mat = np.ones((num_labels, num_labels)) - np.eye(num_labels)
    kfold = 10
    num_perc_lst = [1, 2, 4, 8, 16, 25, 35, 50]
    perc_step = 5
    
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

    # generate data 
    generate_data_pxgl_samples(samples_type, priors, means, covs, label_ids)

    ##theoretical classifier
    calc_theoretical_classifier(samples_type['D100'])

    ## train MLP
    for i, key in enumerate(list(samples_type.keys())[:-1]):
        train_kfoldMLP(samples_type[key], samples_type['D100k'], kfold, num_perc_lst)