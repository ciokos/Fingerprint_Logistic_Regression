from scipy.stats import entropy
import numpy as np
from numpy import unique
import pywt


def get_features(list_values):
    var = np.var(list_values)
    std = np.std(list_values)
    ent = get_entropy(list_values)
    return [std, var, ent]


def get_entropy(image):
    _, counts = unique(image, return_counts=True)
    ent = entropy(counts, base=2)
    return np.array(ent)


def feature_extraction(image):
    features = []
    # https: // www.researchgate.net / publication / 318479953_Fingerprint_based_Automatic_Human_Gender_Identification
    # Discrete Wavelet Transform
    list_coeff = pywt.wavedec2(image, 'db1', level=4)

    features += get_features(list_coeff[0])
    features += get_features(list_coeff[1][0])
    features += get_features(list_coeff[1][1])
    features += get_features(list_coeff[1][2])
    features += get_features(list_coeff[2][0])
    features += get_features(list_coeff[2][1])
    features += get_features(list_coeff[2][2])
    features += get_features(list_coeff[3][0])
    features += get_features(list_coeff[3][1])
    features += get_features(list_coeff[3][2])
    features += get_features(list_coeff[4][0])
    features += get_features(list_coeff[4][1])
    features += get_features(list_coeff[4][2])
    return features


def feature_extraction_dataset(images):
    X = []
    print("starting feature extraction...")
    idx = 1
    count = len(images)
    for image in images:
        X.append(feature_extraction(image))
        if idx % 100 == 0:
            print("done {}%".format(idx * 100 / count))
        idx += 1
    X = np.array(X)
    print("done feature extraction.")
    return X
