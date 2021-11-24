import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def train_val_split(data, idx, kfold):

    num_samples = data.shape[0]
    num_samples_batch = num_samples//kfold
    
    train_data = np.concatenate((data[0:idx*num_samples_batch], data[(idx+1)*num_samples_batch:]), axis=0)
    val_data = data[idx*num_samples_batch : (idx+1)*num_samples_batch]

    return train_data, val_data

def MOS(data, num_gmm_lst, kfold):

    num_samples = data.shape[0]
    gmm_mls = []

    for num_gmm in num_gmm_lst:

        max_likelihoods = []
        for val_idx in range(kfold):

            # train test split
            train_data, val_data = train_val_split(data, val_idx, kfold)

            GMM = GaussianMixture(num_gmm, covariance_type='full', 
                    random_state=0)        

            GMM.fit(train_data)

            max_likelihood = GMM.score(val_data)
            num_val_samples = val_data.shape[0]
            max_likelihoods.append(num_val_samples * max_likelihood)
            
            print('val idx: ', val_idx, ' num_val_samples: ', num_val_samples, ' num_gmm: ', num_gmm, ' mls: ', np.round(max_likelihood, 3))

        mean_max_likelihoods = np.sum(max_likelihoods)/num_samples
        gmm_mls.append(mean_max_likelihoods)
        print('num_gmm: ', num_gmm, ' mean_mle: ',np.round(mean_max_likelihoods, 3))

    desired_num_gmm = num_gmm_lst[np.argmax(gmm_mls)]

    return desired_num_gmm

def get_feature_vector(img):

    h, w = img.shape[:2]
    num_pixels = h * w
    
    feat_vec = np.zeros((num_pixels, 5), dtype='float') 

    for row in range(h):
        for col in range(w):
            feat_vec[row*w + col, 0] = row
            feat_vec[row*w + col, 1] = col
            feat_vec[row*w + col, 2] = img[row, col, 2] #red
            feat_vec[row*w + col, 3] = img[row, col, 1] #green
            feat_vec[row*w + col, 4] = img[row, col, 0] #blue

    norm_feat_vec = normalize(feat_vec, axis=0, norm='max')

    return norm_feat_vec

if __name__ == "__main__":

    img_path = '42049.jpg'

    num_gmm_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    kfold = 10

    img = cv2.imread(img_path)  

    print('img shape ',img.shape)
    norm_feat_vec = get_feature_vector(img)
    print('norm_feat_vec: ',norm_feat_vec)
    
    desired_num_gmm = MOS(norm_feat_vec, num_gmm_lst, kfold)

    print('desired_num_gmm: ',desired_num_gmm)    

    model = GaussianMixture(desired_num_gmm, covariance_type='full', 
                    random_state=0, init_params='kmeans', max_iter=100)            
    model.fit(norm_feat_vec)

    prediction = np.zeros((norm_feat_vec.shape[0], desired_num_gmm))
    for i in range(desired_num_gmm):
        pdf = multivariate_normal.pdf(norm_feat_vec, mean=model.means_[i,:],cov=model.covariances_[i,:,:])
        prediction[:, i] = model.weights_[i] * pdf

    prediction = np.argmax(prediction, axis=1)
    print('prediction ', prediction,prediction.shape)
    prediction = prediction.reshape((img.shape[0], img.shape[1]))
    plt.imshow(prediction)
    plt.show()