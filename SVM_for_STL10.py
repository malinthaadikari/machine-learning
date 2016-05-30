import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import datetime
from sklearn.externals import joblib

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = '/home/malintha/projects/ML/projects/Recognizor/stl10_binary'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = '/home/malintha/projects/ML/projects/Recognizor/stl10_binary/train_X.bin'

# path to the binary train file with labels
LABEL_PATH = '/home/malintha/projects/ML/projects/Recognizor/stl10_binary/train_y.bin'

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


if __name__ == "__main__":

    images = read_all_images(DATA_PATH)


    def train_and_evaluate(clf, train_x, train_y):
        clf.fit(train_x, train_y)


    my_svc = SVC(kernel='linear')

    nsamples, nx, ny, nz = images.shape
    reshaped_train_dataset = images.reshape((nsamples, nx * ny * nz))

    X_train, X_test, Y_train, Y_test = train_test_split(reshaped_train_dataset, read_labels(LABEL_PATH), test_size=0.20,
                                                        random_state=33)

    train_and_evaluate(my_svc, X_train, Y_train)

    # Persisting trained model for future predicting purposes
    filename = '/tmp/digits_classifier.joblib.pkl'
    _ = joblib.dump(my_svc, filename, compress=9)

    persisted_classifier = joblib.load(filename)

    print(metrics.accuracy_score(Y_test, persisted_classifier.predict(X_test)))