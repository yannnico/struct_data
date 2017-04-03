import cv2
import numpy as np
import os
import h5py
from sklearn.svm import SVR

SZ = 20
bin_n = 16  # Number of bins


def import_data():
    images = {}
    labels = {}

    for files in os.listdir("data/cows-images"):
        key = files.split('-')[1]
        images[key] = cv2.imread("data/cows-images/" + files)

    for files in os.listdir("data/cows-labels"):
        key = files.split('-')[1]
        with open('data/cows-labels/' + files) as input_file:

            for line in input_file:
                if 'Bounding' in line:
                    item = line[70:-2]
                    values = []
                    for elem in item.split('-'):
                        elem = elem.strip()
                        elem = elem.strip(')')
                        elem = elem.strip('(')
                        values.extend(elem.split(','))

        x_min = int(values[0])
        y_min = int(values[1])
        x_max = int(values[2])
        y_max = int(values[3])

        labels[key] = np.array([x_min, y_min, x_max, y_max])

    print(images.keys())

    h5f = h5py.File("data/images_labels.h5", 'w')

    for key in images.keys():
        if key in labels.keys():
            print(key)
            h5f.create_dataset('image_' + key, data=images[key])
            h5f.create_dataset('labels_' + key, data=labels[key])
    h5f.close()


def from_image_and_labels_to_dataset():
    images = {}
    labels = {}

    h5f = h5py.File("data/images_labels.h5", 'r')

    keys = h5f.keys()

    for key in keys:
        name = key.split('_')
        if name[0] == 'image':
            images[name[1]] = np.array(h5f[key])
        if name[0] == 'labels':
            labels[name[1]] = np.array(h5f[key])
    h5f.close()

    print(images.keys())

    neg_labels = {}

    for key in images.keys():
        if key in labels.keys():
            print(key)
            sh = np.shape(images[key])
            # print(sh)
            # print(labels[key])

            left = (labels[key][0] > (sh[1] - labels[key][2]))
            top = (labels[key][1] > (sh[0] - labels[key][3]))

            # print(left)
            # print(top)

            if left and labels[key][0] > (labels[key][2] - labels[key][0]):
                #  pick random image on the left (same size)
                x_min = int(labels[key][0] / 2 - (labels[key][2] - labels[key][0]) / 2)
                y_min = int(max(0, labels[key][1] + (np.random.rand() - 0.5) * 10))
                x_max = int(labels[key][0] / 2 + (labels[key][2] - labels[key][0]) / 2)
                y_max = int(min(sh[0], labels[key][3] + (np.random.rand() - 0.5) * 10))
            elif not left and (sh[1] - labels[key][2]) > (labels[key][2] - labels[key][0]):
                #  pick random image on the right (same size)
                x_min = int((sh[1] + labels[key][2]) / 2 - (labels[key][2] - labels[key][0]) / 2)
                y_min = int(max(0, labels[key][1] + (np.random.rand() - 0.5) * 10))
                x_max = int((sh[1] + labels[key][2]) / 2 + (labels[key][2] - labels[key][0]) / 2)
                y_max = int(min(sh[0], labels[key][3] + (np.random.rand() - 0.5) * 10))
            else:
                print("_________here_________")
                x = int(labels[key][2] - labels[key][0] - (np.random.rand()) * 10)
                y = int(labels[key][3] - labels[key][1] - (np.random.rand()) * 10)

                if left:
                    x_min = int((np.random.rand()) * 10)
                    x_max = x
                else:
                    x_min = sh[1] - x
                    x_max = int(sh[1] - (np.random.rand()) * 10)

                if top:
                    y_min = int((np.random.rand()) * 10)
                    y_max = y
                else:
                    y_min = sh[0] - y
                    y_max = int(sh[0] - (np.random.rand()) * 10)

            neg_labels[key] = np.array([x_min, y_min, x_max, y_max])

    data = []
    labels_list = []
    for key in images.keys():
        pos_img = hog(images[key][labels[key][0]:labels[key][2], labels[key][1]:labels[key][3]])
        neg_img = hog(images[key][neg_labels[key][0]:neg_labels[key][2], neg_labels[key][1]:neg_labels[key][3]])
        data.append(pos_img)
        data.append(neg_img)
        labels_list.append(1)
        labels_list.append(-1)

    data = np.array(data)
    labels_list = np.array(labels_list)

    h5f = h5py.File("dataset.h5", 'w')

    h5f.create_dataset("data", data=data)
    h5f.create_dataset("labels", data=labels_list)

    h5f.close()


def load_data():
    h5f = h5py.File("dataset.h5", 'r')

    data = np.array(h5f['data'])
    labels = np.array(h5f['labels'])

    h5f.close()

    print(np.shape(data))
    print(np.shape(labels))

    x_train = data[:100]
    x_test = data[100:]
    y_train = labels[:100]
    y_test = labels[100:]

    clf = SVR(kernel='linear', C=1.0, epsilon=0.2)
    clf.fit(x_train, y_train)

    print(clf.predict(x_test))
    print(y_test)


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


if __name__ == "__main__":
    # import_data()
    # from_image_and_labels_to_dataset()
    load_data()
