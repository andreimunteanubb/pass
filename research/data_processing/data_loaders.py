import os
import random
import time
from pathlib import Path

from scipy import ndimage as ndi

import numpy as np
from imutils import paths
import cv2
from skimage import morphology
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split


def dataset_split():
    dirname = os.path.dirname(__file__)
    data_dir = Path('../input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]')
    data_dir = Path(dirname, data_dir)
    data_list = sorted(list(paths.list_images(data_dir)))

    random.seed(88)
    random.shuffle(data_list)

    train_list, test_list = train_test_split(data_list, train_size=0.90, shuffle=True, random_state=88)

    print('number of testing list -:', len(test_list))
    print('number of training list-:', len(train_list))

    print('Number of samples in dataset:',
          len(list(data_list)), '\n')

    print('Number of samples in each class:', '\n')
    print("#1 Benign ---------------:",
          len(list(paths.list_images(Path(data_dir, "Benign")))))
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


def read_data(samples_list: list, dir_name: str, is_train=False):
    p = 0
    dirname = os.path.dirname(__file__)
    crr_dir = Path(dirname)
    tic = time.perf_counter()
    for img in samples_list[:]:

        i = cv2.imread(img)
        i = cv2.resize(i, (224, 224))
        lable = img.split(os.path.sep)[-2]

        if (lable == "Benign"):
            b = (Path(crr_dir, f'../tmp/{dir_name}/benign/' + lable + str(p) + '.png'))
        if (lable == "[Malignant] Pre-B"):
            b = (Path(crr_dir, f'../tmp/{dir_name}/PreB/' + lable + str(p) + '.png'))
        if (lable == "[Malignant] Pro-B"):
            b = (Path(crr_dir, f'../tmp/{dir_name}/ProB/' + lable + str(p) + '.png'))
        if (lable == "[Malignant] early Pre-B"):
            b = (Path(crr_dir, f'../tmp/{dir_name}/EarlyPreB/' + lable + str(p) + '.png'))
        p += 1
        os.makedirs(b.parent.as_posix(), exist_ok=True)
        cv2.imwrite(b.as_posix(), i)

        if is_train:
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
                b = (Path(crr_dir, f'../tmp/{dir_name}/benign/' + lable + str(p) + '.png'))
            if (lable == "[Malignant] Pre-B"):
                b = (Path(crr_dir, f'../tmp/{dir_name}/PreB/' + lable + str(p) + '.png'))
            if (lable == "[Malignant] Pro-B"):
                b = (Path(crr_dir, f'../tmp/{dir_name}/ProB/' + lable + str(p) + '.png'))
            if (lable == "[Malignant] early Pre-B"):
                b = (Path(crr_dir, f'../tmp/{dir_name}/EarlyPreB/' + lable + str(p) + '.png'))
            p += 1
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            os.makedirs(b.parent.as_posix(), exist_ok=True)
            cv2.imwrite(b.as_posix(), out)

    toc2 = time.perf_counter()
    # print(f"2917 samples processed in {((toc2 - tic) / 60)} minutes")
