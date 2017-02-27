import numpy as np
import random

def dequeue_and_enqueue(items, n):
    return np.concatenate(np.split(items, [-n], axis=0)[::-1])

def shuffle(images, labels):

    c = list(zip(images, labels))

    random.shuffle(c)

    images, labels = zip(*c)
    return images, labels

def prepare(data):
    images, labels = shuffle(data['sequences'], data['labels'])
    print(len(images))
    print(len(labels))
    return np.array(images), np.array(labels)[:, 0].reshape(len(labels), 1)
