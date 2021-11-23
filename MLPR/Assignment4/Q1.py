import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def gen_data()