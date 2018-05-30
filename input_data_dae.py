# coding=utf-8
import tensorflow as tf
import numpy as np
import tifffile as tiff
import pickle
import random

flags = tf.flags

flags.DEFINE_string('FILE_2015', '/home/zyt/data/TianChi/preliminary/quickbird2015.tif',
                    'the directory to quickbird2015.tif')
flags.DEFINE_string('FILE_2017', '/home/zyt/data/TianChi/preliminary/quickbird2017.tif',
                    'the directory to quickbird2017.tif')
flags.DEFINE_string('FILE_cadastral2015', '/home/zyt/data/TianChi/20170907_hint/cadastral2015.tif',
                    'the directory to cadastral2015.tif')
flags.DEFINE_string('FILE_tinysample', '/home/zyt/data/TianChi/20170907_hint/tinysample.tif',
                    'the directory to tinysample.tif')

FLAGS = flags.FLAGS

FILE_2015 = '/home/zyt/data/TianChi/preliminary/quickbird2015.tif'
FILE_2017 = '/home/zyt/data/TianChi/preliminary/quickbird2017.tif'
FILE_cadastral2015 = '/home/zyt/data/TianChi/20170907_hint/cadastral2015.tif'
FILE_tinysample = '/home/zyt/data/TianChi/20170907_hint/tinysample.tif'
FILE_label = '/home/zyt/data/TianChi/label/label1011.tif'

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
    mins = np.percentile(matrix, 0, axis=0)
    maxs = np.percentile(matrix, 100, axis=0)
    lens = maxs - mins
    if np.min(lens) == 0:
        return np.reshape(matrix, [w, h, d])

    matrix = (matrix - mins[None, :]) / lens[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def build_dataset(image_size=16, channels=4):
    im_2015 = np.array(tiff.imread(FILE_2015).transpose([1, 2, 0]), dtype=np.float32)
    im_2017 = np.array(tiff.imread(FILE_2017).transpose([1, 2, 0]), dtype=np.float32)

    im_tiny = np.array(tiff.imread(FILE_tinysample), dtype=np.float32)
    im_label = np.array(tiff.imread(FILE_label), dtype=np.float32)
    im_cada = np.array(tiff.imread(FILE_cadastral2015), dtype=np.float32)

    train_2015 = np.zeros([100000, image_size, image_size, channels], dtype=np.float32)
    train_2017 = np.zeros([100000, image_size, image_size, channels], dtype=np.float32)
    test_2015 = np.zeros([20000, image_size, image_size, channels], dtype=np.float32)
    test_2017 = np.zeros([20000, image_size, image_size, channels], dtype=np.float32)

    train_len = 0
    test_len = 0
    for xstart, xend in zip(range(0, im_tiny.shape[0]-1, 16), range(image_size, im_tiny.shape[0], 16)):
        for ystart, yend in zip(range(0, im_tiny.shape[1]-1, 16), range(image_size, im_tiny.shape[1], 16)):
            tmp_tiny = im_tiny[xstart:xend, ystart:yend, 0]
            tmp_label = im_label[xstart:xend, ystart:yend]
            tmp_cada = im_cada[xstart:xend, ystart:yend]

            if np.max(tmp_label) == 0 and np.max(tmp_tiny) == 0 and np.max(tmp_cada) == 0:
                tmp_2015 = im_2015[xstart:xend, ystart:yend, :]
                tmp_2017 = im_2017[xstart:xend, ystart:yend, :]
                if np.max(tmp_2015) == 0 or np.max(tmp_2017) == 0:
                    continue

                if train_len < 100000:
                    train_2015[train_len] = scale_percentile(tmp_2015)
                    train_2017[train_len] = scale_percentile(tmp_2017)
                    train_len += 1
                elif test_len < 20000:
                    test_2015[test_len] = scale_percentile(tmp_2015)
                    test_2017[test_len] = scale_percentile(tmp_2017)
                    test_len += 1
                else:
                    break
        if train_len >= 100000 and test_len >= 20000:
            print("build dataset successfully!")
            break
        else:
            print("at now(x is %d), train_len is %d, test_len is %d" % (xstart, train_len, test_len))

    del im_2015
    del im_2017
    del im_cada
    del im_tiny

    output_file = open("/home/zyt/data/TianChi/dataset/20171023/train_2015_16*16.pkl", "wb")
    pickle.dump(train_2015, output_file)
    output_file.close()
    del train_2015
    print("save train_2015 successfully!")
    output_file = open("/home/zyt/data/TianChi/dataset/20171023/train_2017_16*16.pkl", "wb")
    pickle.dump(train_2017, output_file)
    output_file.close()
    del train_2017
    print("save train_2017 successfully!")
    output_file = open("/home/zyt/data/TianChi/dataset/20171023/test_2015_16*16.pkl", "wb")
    pickle.dump(test_2015, output_file)
    output_file.close()
    del test_2015
    print("save test_2015 successfully!")
    output_file = open("/home/zyt/data/TianChi/dataset/20171023/test_2017_16*16.pkl", "wb")
    pickle.dump(test_2017, output_file)
    output_file.close()
    print("save test_2017 successfully!")


def dae_iterator(batch_size):
    input_file = open("/home/zyt/data/TianChi/dataset/20171023/train_2015_16*16.pkl", "rb")
    train_2015 = pickle.load(input_file)
    input_file.close()
    input_file = open("/home/zyt/data/TianChi/dataset/20171023/train_2017_16*16.pkl", "rb")
    train_2017 = pickle.load(input_file)
    input_file.close()

    epoch_size = len(train_2015) // batch_size
    ids = range(epoch_size)
    random.shuffle(ids)

    for i in ids:
        batch_x = train_2015[i*batch_size:(i+1)*batch_size]
        batch_y = train_2017[i*batch_size:(i+1)*batch_size]
        yield batch_x, batch_y


def test_data():
    input_file = open("/home/zyt/data/TianChi/dataset/20171023/test_2015_16*16.pkl", "rb")
    test_2015 = pickle.load(input_file)
    input_file.close()
    input_file = open("/home/zyt/data/TianChi/dataset/20171023/test_2017_16*16.pkl", "rb")
    test_2017 = pickle.load(input_file)
    input_file.close()

    return test_2015[:100], test_2017[:100]


def dae_iterator_32(batch_size):
    input_file = open("/home/zyt/data/TianChi/dataset/20170916/train_2015.pkl", "rb")
    train_2015 = pickle.load(input_file)
    input_file.close()
    input_file = open("/home/zyt/data/TianChi/dataset/20170916/train_2017.pkl", "rb")
    train_2017 = pickle.load(input_file)
    input_file.close()

    epoch_size = len(train_2015) // batch_size
    ids = range(epoch_size)
    random.shuffle(ids)

    for i in ids:
        batch_x = train_2015[i*batch_size:(i+1)*batch_size]
        batch_y = train_2017[i*batch_size:(i+1)*batch_size]
        yield batch_x, batch_y


def test_data_32():
    input_file = open("/home/zyt/data/TianChi/dataset/20170916/test_2015.pkl", "rb")
    test_2015 = pickle.load(input_file)
    input_file.close()
    input_file = open("/home/zyt/data/TianChi/dataset/20170916/test_2017.pkl", "rb")
    test_2017 = pickle.load(input_file)
    input_file.close()

    return test_2015[:100], test_2017[:100]


if __name__ == '__main__':
    build_dataset()
