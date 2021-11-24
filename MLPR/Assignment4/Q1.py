import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras.models import Sequential
from keras.layers import Dense
np.set_printoptions(suppress=True)
from keras.optimizers import SGD
from sklearn.svm import SVC

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

def get_model(first_num_nodes, num_labels=1):

    sgd = SGD(lr=0.05, momentum=0.9)

    model = Sequential()

    # first layer
    fc1_act = Dense(units = first_num_nodes, kernel_initializer = 'random_uniform', activation = 'relu')
    model.add(fc1_act)

    # Second layer
    fc2_act = Dense(units = num_labels, kernel_initializer = 'random_uniform', activation = 'sigmoid')
    model.add(fc2_act)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics = ['accuracy'])

    return model

def get_hp_values(num):

    hp_values = np.meshgrid(np.geomspace(0.05, 10, num), np.geomspace(0.05, 20, num))    
    hp_values[0] = hp_values[0].reshape(num*num)
    hp_values[1] = hp_values[1].reshape(num*num)
    hp_values = np.vstack((hp_values[0], hp_values[1])).T

    return hp_values

def SVC_hyperparams(data_wt_labels, kfold):

    num_samples = data_wt_labels.shape[1] #(3, N)
    data_wt_labels = data_wt_labels[:, np.random.permutation(data_wt_labels.shape[1])] #shuffle

    data = data_wt_labels[:2,:].T #(N, 2)
    labels = data_wt_labels[2,:].T

    print('data shape: ',data.shape)
    print('labels shape: ',labels.shape)

    num = 20
    hp_values = get_hp_values(num)
    hp_accs = np.zeros((num*num, 2), dtype='float')

    perc_lst = []
    for C, kernel_width in hp_values:

        err_lst = []
        acc_lst = []
        skf = StratifiedKFold(n_splits=kfold, shuffle=False)

        for(val_idx, (train, val)) in enumerate(skf.split(data, labels)):

            # train
            train_data = data[train]
            train_labels = labels[train]

            # val
            val_data = data[val]
            val_labels = labels[val]

            # get model
            gamma = 1/(2*kernel_width**2)
            model = SVC(C=C, kernel='rbf', gamma=gamma)

            # train
            model.fit(train_data, train_labels, batch_size = 100, epochs = 300, verbose=0)
            
            # validate
            (err, accuracy) = model.evaluate(val_data, val_labels)

            print('num_samples:', num_samples,' num_perc: ',num_perc,' val idx: ', val_idx, ' error: ', np.round(err, 4), ' accuracy: ', np.round(accuracy, 4))
            err_lst.append(err)
            acc_lst.append(accuracy)

        mean_err = np.mean(np.array(err_lst))
        std_err = np.std(np.array(err_lst))
        mean_acc = np.mean(np.array(acc_lst))

        print('num_samples:', num_samples, ' num_perc: ',num_perc,' mean error: ', np.round(mean_err, 4), ' std error: ',np.round(std_err, 4), ' mean_acc: ', mean_acc)
        perc_lst.append(mean_err)

    perc_lst = np.array(perc_lst)
    print('pe for each perceptron: ', perc_lst)
    desired_num_perc = num_perc_lst[np.argmin(perc_lst)]

    return desired_num_perc

def MLP_hyperparams(data_wt_labels, kfold, num_perc_lst):

    num_samples = data_wt_labels.shape[1] #(3, N)
    data_wt_labels = data_wt_labels[:, np.random.permutation(data_wt_labels.shape[1])] #shuffle

    data = data_wt_labels[:2,:].T #(N, 2)
    labels = data_wt_labels[2,:].T

    print('data shape: ',data.shape)
    print('labels shape: ',labels.shape)

    perc_lst = []
    for num_perc in num_perc_lst:

        err_lst = []
        acc_lst = []
        skf = StratifiedKFold(n_splits=kfold, shuffle=False)

        for(val_idx, (train, val)) in enumerate(skf.split(data, labels)):

            # train
            train_data = data[train]
            train_labels = labels[train]

            # val
            val_data = data[val]
            val_labels = labels[val]

            #data shape summary
            # print('train data shape ',train_data.shape)
            # print('train label shape ',train_labels.shape)
            # print('val data shape ',val_data.shape)
            # print('val labels shape ',val_labels.shape)

            # get model
            model = get_model(num_perc)

            # train
            model.fit(train_data, train_labels, batch_size = 100, epochs = 300, verbose=0)
            
            # validate
            (err, accuracy) = model.evaluate(val_data, val_labels)

            print('num_samples:', num_samples,' num_perc: ',num_perc,' val idx: ', val_idx, ' error: ', np.round(err, 4), ' accuracy: ', np.round(accuracy, 4))
            err_lst.append(err)
            acc_lst.append(accuracy)

        mean_err = np.mean(np.array(err_lst))
        std_err = np.std(np.array(err_lst))
        mean_acc = np.mean(np.array(acc_lst))

        print('num_samples:', num_samples, ' num_perc: ',num_perc,' mean error: ', np.round(mean_err, 4), ' std error: ',np.round(std_err, 4), ' mean_acc: ', mean_acc)
        perc_lst.append(mean_err)

    perc_lst = np.array(perc_lst)
    print('pe for each perceptron: ', perc_lst)
    desired_num_perc = num_perc_lst[np.argmin(perc_lst)]

    return desired_num_perc

def train_kfoldMLP(train_wt_cls, test_wt_cls, kfold):

    num_train = train_wt_cls.shape[1]
    
    train_wt_cls = train_wt_cls[:, np.random.permutation(train_wt_cls.shape[1])] #shuffle
    train_data = train_wt_cls[:2,:].T #(N, 2)
    train_labels = train_wt_cls[2,:].T

    num_perc_lst = np.array([3, 4])*2 #np.array([3, 4, 5, 6, 7, 8, 9, 10])*2

    print('train_wt_cls shape ',train_wt_cls.shape)
    # Model Order Selection
    desired_num_perc = MLP_hyperparams(train_wt_cls, kfold, num_perc_lst)

    # get model
    model = get_model(desired_num_perc)
    
    # train
    model.fit(train_data, train_labels, batch_size = 100, epochs = 300, verbose=0)

    print('model summary')
    print(model.summary())

    # validate
    test_data = test_wt_cls[:2,:].T #(N, 2)
    test_labels = test_wt_cls[2,:].T
    (val_err, val_acc) = model.evaluate(test_data, test_labels)

    print('num_samples: ',num_train,' desired_num_perc: ',desired_num_perc,' val_err: ', val_err, ' val_acc: ', val_acc)

def train_kfoldSVC(train_wt_cls, test_wt_cls, kfold):

    num_train = train_wt_cls.shape[1]
    
    train_wt_cls = train_wt_cls[:, np.random.permutation(train_wt_cls.shape[1])] #shuffle
    train_data = train_wt_cls[:2,:].T #(N, 2)
    train_labels = train_wt_cls[2,:].T

    # Model Order Selection
    desired_num_perc = SVC_hyperparams(train_wt_cls, kfold, num_perc_lst)

    # get model
    model = get_model(desired_num_perc)
    
    # train
    model.fit(train_data, train_labels, batch_size = 100, epochs = 300, verbose=0)

    print('model summary')
    print(model.summary())

    # validate
    test_data = test_wt_cls[:2,:].T #(N, 2)
    test_labels = test_wt_cls[2,:].T
    (val_err, val_acc) = model.evaluate(test_data, test_labels)

    print('num_samples: ',num_train,' desired_num_perc: ',desired_num_perc,' val_err: ', val_err, ' val_acc: ', val_acc)

def plot_prediction(model, test_data, test_labels):

    pass

    

if __name__ == "__main__":

    priors = [0.6, 0.4]
    label_ids = [0, 1]
    num_train_samples = 1000
    num_test_samples = 10000
    noise_mu = [0, 0]
    noise_sigma = np.eye(2, dtype=float)
    r = [2, 4]
    kfold = 10
    
    ## train
    train_wt_cls = gen_data(num_train_samples, priors)
    #plot_data(train_wt_cls, label_ids)

    ## test
    test_wt_cls = gen_data(num_test_samples, priors)
    #plot_data(test_wt_cls, label_ids)

    ## Train MLP
    train_kfoldMLP(train_wt_cls, test_wt_cls, kfold)