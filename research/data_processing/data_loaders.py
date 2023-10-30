import os
from datetime import time
from random import random
from scipy import ndimage as ndi

import numpy as np
from imutils import paths
import cv2
from scipy.ndimage import morphology
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split


def dataset_split():
    data_dir = '../input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]'
    data_list = sorted(list(paths.list_images(data_dir)))

    random.seed(88)
    random.shuffle(data_list)

    train_list, test_list = train_test_split(data_list, train_size=0.90, shuffle=True, random_state=88)

    print('number of testing list -:', len(test_list))
    print('number of training list-:', len(train_list))

    print('Number of samples in dataset:',
          len(list(paths.list_images("../input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]"))), '\n')

    print('Number of samples in each class:', '\n')
    print("#1 Benign ---------------:",
          len(list(paths.list_images("../input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]/Benign"))))
    print("#2 Malignant[Early PreB] :", len(list(
        paths.list_images("../input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]/[Malignant] early Pre-B"))))
    print("#3 Malignant[PreB] ------:", len(list(
        paths.list_images("../input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]/[Malignant] Pre-B"))))
    print("#4 Malignant[ProB] ------:", len(list(
        paths.list_images("../input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]/[Malignant] Pro-B"))))

    return train_list, test_list

def read_test_data(test_list):
    p = 0

    for img in test_list[:]:

        i = cv2.imread(img)
        i = cv2.resize(i, (224, 224))
        lable = img.split(os.path.sep)[4]

        if (lable == "Benign"):
            b = ('/tmp/prepared_test/benign/' + lable + str(p) + '.png')
        if (lable == "[Malignant] Pre-B"):
            b = ('/tmp/prepared_test/PreB/' + lable + str(p) + '.png')
        if (lable == "[Malignant] Pro-B"):
            b = ('/tmp/prepared_test/ProB/' + lable + str(p) + '.png')
        if (lable == "[Malignant] early Pre-B"):
            b = ('/tmp/prepared_test/EarlyPreB/' + lable + str(p) + '.png')
        p += 1
        cv2.imwrite(b, i)

def read_training_data(train_list):
    p = 0
    tic = time.perf_counter()

    for img in train_list[:]:

        i = cv2.imread(img)
        i = cv2.resize(i, (224, 224))
        lable = img.split(os.path.sep)[4]

        if (lable == "Benign"):
            b = ('/tmp/prepared_data/benign/' + lable + str(p) + '.png')
        if (lable == "[Malignant] Pre-B"):
            b = ('/tmp/prepared_data/PreB/' + lable + str(p) + '.png')
        if (lable == "[Malignant] Pro-B"):
            b = ('/tmp/prepared_data/ProB/' + lable + str(p) + '.png')
        if (lable == "[Malignant] early Pre-B"):
            b = ('/tmp/prepared_data/EarlyPreB/' + lable + str(p) + '.png')
        p += 1
        cv2.imwrite(b, i)

        # -------- Segmentation ---------
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i_lab = cv2.cvtColor(i, cv2.COLOR_RGB2LAB)  # RGB -> LAB
        l, a, b = cv2.split(i_lab)
        i2 = a.reshape(a.shape[0] * a.shape[1], 1)
        km = KMeans(n_clusters=7, random_state=0).fit(i2)  # Clustring
        p2s = km.cluster_centers_[km.labels_]
        ic = p2s.reshape(a.shape[0], a.shape[1])
        ic = ic.astype(np.uint8)
        r, t = cv2.threshold(ic, 141, 255, cv2.THRESH_BINARY)  # Binary Thresholding
        fh = ndi.binary_fill_holes(t)  # fill holes
        m1 = morphology.remove_small_objects(fh, 200)
        m2 = morphology.remove_small_holes(m1, 250)
        m2 = m2.astype(np.uint8)
        out = cv2.bitwise_and(i, i, mask=m2)

        if (lable == "Benign"):
            b = ('/tmp/prepared_data/benign/' + lable + str(p) + '.png')
        if (lable == "[Malignant] Pre-B"):
            b = ('/tmp/prepared_data/PreB/' + lable + str(p) + '.png')
        if (lable == "[Malignant] Pro-B"):
            b = ('/tmp/prepared_data/ProB/' + lable + str(p) + '.png')
        if (lable == "[Malignant] early Pre-B"):
            b = ('/tmp/prepared_data/EarlyPreB/' + lable + str(p) + '.png')
        p += 1
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(b, out)

    toc2 = time.perf_counter()
    print(f"2917 samples processed in {((toc2 - tic) / 60)} minutes")