import tensorflow as tf
import numpy as np
import tifffile as tiff

FILE_2015 = '/home/zyt/data/TianChi/preliminary/quickbird2015.tif'
FILE_2017 = '/home/zyt/data/TianChi/preliminary/quickbird2017.tif'
FILE_cadastral2015 = '/home/zyt/data/TianChi/20170907_hint/cadastral2015.tif'
FILE_tinysample = '/home/zyt/data/TianChi/20170907_hint/tinysample.tif'

# im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
# im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
# im_tiny = tiff.imread(FILE_tinysample)
# im_cada = tiff.imread(FILE_cadastral2015)

# im_2015.shape  (5106, 15106, 4)
# im_2017.shape  (5106, 15106, 4)
# im_tiny.shape  (5106, 15106, 3)
# im_cada.shape  (5106, 15106)


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w*h, d]).astype(np.float32)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0)
    lens = maxs - mins
    matrix = (matrix - mins[None, :]) / lens[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def bulid_dataset(xstart=0, xend=5106, ystart=0, yend=15106):
    im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
    im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
    im_tiny = tiff.imread(FILE_tinysample)

    data_2015 = np.array(im_2015[xstart:xend, ystart:yend, :], dtype=np.float32)
    data_2015 = scale_percentile(data_2015)

    data_2017 = np.array(im_2017[xstart:xend, ystart:yend, :], dtype=np.float32)
    data_2017 = scale_percentile(data_2017)

    inputs = data_2017 - data_2015
    inputs = np.reshape(inputs, [-1, 4])

    data_tiny = np.array(im_tiny[xstart:xend, ystart:yend, 0], dtype=np.int64)
    data_tiny = np.reshape(data_tiny, [-1, 1])
    data_tiny[data_tiny > 0] = 1

    labels = np.zeros(shape=(len(data_tiny), 2))
    for i in range(len(data_tiny)):
        labels[i, data_tiny[i]] = 1

    return inputs, labels


def bulid_dataset_singleChannel(xstart=0, xend=5106, ystart=0, yend=15106):
    im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
    im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
    im_tiny = tiff.imread(FILE_tinysample)

    data_2015 = np.array(im_2015[xstart:xend, ystart:yend, 2:], dtype=np.float32)

    data_2017 = np.array(im_2017[xstart:xend, ystart:yend, 2:], dtype=np.float32)

    inputs = data_2017 - data_2015
    inputs = np.reshape(inputs, [-1, 2])

    data_tiny = np.array(im_tiny[xstart:xend, ystart:yend, 0], dtype=np.int32)
    data_tiny = np.reshape(data_tiny, [-1, 1])
    data_tiny[data_tiny > 0] = 1

    labels = np.zeros(shape=(len(data_tiny), 2))
    for i in range(len(data_tiny)):
        labels[i, data_tiny[i]] = 1

    return inputs, labels


