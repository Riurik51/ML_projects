#!/usr/bin/env python3

import os
import array

import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))

IMAGES_TEST = os.path.join(PATH, 't10k-images.idx3-ubyte')
LABELS_TEST = os.path.join(PATH, 't10k-labels.idx1-ubyte')

IMAGES_TRAIN = os.path.join(PATH, 'train-images.idx3-ubyte')
LABELS_TRAIN = os.path.join(PATH, 'train-labels.idx1-ubyte')


def read_label_file(filename, n_objects=None, offset=0):
    magic = 2049

    with open(filename, 'rb') as f:
        buffer = array.array('I', f.read(8))
        if buffer[0] != magic:
            buffer.byteswap()
        magic, n_items = buffer

        n_size = n_objects if n_objects else n_items - offset
        if offset >= n_items:
            raise IndexError('Out of bounds!')
        elif offset + n_size > n_items:
            n_size = n_items - offset

        f.seek(offset, 1)
        labels = array.array('B', f.read(n_size))

    return np.asarray(labels, dtype=np.uint8)


def read_image_file(filename, n_objects=None, offset=0):
    magic = 2051

    with open(filename, 'rb') as f:
        buffer = array.array('I', f.read(16))
        if buffer[0] != magic:
            buffer.byteswap()
        magic, n_items, n_rows, n_cols = buffer

        n_size = n_objects if n_objects else n_items - offset
        if offset >= n_items:
            raise IndexError('Out of bounds!')
        elif offset + n_size > n_items:
            n_size = n_items - offset

        f.seek(offset * n_cols * n_rows, 1)
        size = n_size * n_cols * n_rows
        buffer = f.read(size)
        images = np.asarray(array.array('B', buffer)).reshape(-1, n_cols, n_rows)

    return np.asarray(images, dtype=np.uint8)


if __name__ == '__main__':
    batch_size, offset = 100, 15
    labels = read_label_file(LABELS_TRAIN, batch_size, offset)
    images = read_image_file(IMAGES_TRAIN, batch_size, offset)

    print(labels)
    n, m = 10, 10
    f, axarr = plt.subplots(n, m, figsize=(6, 6))
    for i in range(n):
        for j in range(m):
            axarr[i, j].imshow(images[m * i + j], cmap='gray_r')
            # axarr[i, j].set_title(labels[m * i + j])
            axarr[i, j].tick_params(
                axis='both', which='both',
                bottom='off', top='off', right='off', left='off',
                labelbottom='off', labelleft='off'
            )
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            # axarr[i, j].axis('off')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()
