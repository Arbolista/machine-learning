from six.moves import cPickle as pickle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import cv2
import os

class Preprocess:

  def __init__(self):
    pass

  def preprocess_and_save_or_load_images(self):
    filename = 'data/preprocessed_images.pickle'
    if os.path.exists(filename):
      return self.load_file(filename)
    else:
      mnist = input_data.read_data_sets("MNIST_data/")
      data = {}

      trainset = mnist.train.images
      data['trainset'] = np.apply_along_axis((lambda image: self.apply_random_transforms_single_image(image)), -1, trainset).reshape(trainset.shape[0], 28, 28)
      data['train_labels'] = mnist.train.labels

      validset = mnist.validation.images
      print(validset.shape)
      print(validset.__class__)
      data['validset'] = np.apply_along_axis((lambda image: self.apply_random_transforms_single_image(image)), -1, validset).reshape(validset.shape[0], 28, 28)
      data['valid_labels'] = mnist.validation.labels

      testset = mnist.test.images
      data['testset'] = np.apply_along_axis((lambda image: self.apply_random_transforms_single_image(image)), -1, testset).reshape(testset.shape[0], 28, 28)
      data['test_labels'] = mnist.test.labels
      with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return data

  def apply_random_transforms_single_image(self, image):
    image = image.reshape(28, 28)
    n_transforms = 3.
    rand_transform = np.random.rand()

    if rand_transform < 1/n_transforms:
      return self.rotate(image).reshape(784,)

    elif rand_transform < 2/n_transforms:
      return self.translate(image).reshape(784,)

    else:
      return self.scale(image).reshape(784,)

  def preprocess_and_save_or_load_sequences(self):
    image_data = self.preprocess_and_save_or_load_images();
    self.do_preprocess_and_save_or_load_sequences('data/test_sequences.pickle', image_data['testset'], image_data['test_labels'])
    self.do_preprocess_and_save_or_load_sequences('data/valid_sequences.pickle', image_data['validset'], image_data['valid_labels'])
    for i in range(len(image_data['trainset'])/5000):
      low = i * 5000
      hi = (i+1) * 5000
      image_set = image_data['trainset'][low:hi]
      label_set = image_data['train_labels'][low:hi]
      padded_i = '{:02d}'.format(i)
      self.do_preprocess_and_save_or_load_sequences('data/train_sequences'+padded_i+'.pickle', image_set, label_set)

  def load_file(self, filename):
    with open(filename, 'rb') as f:
      return pickle.load(f)

  def do_preprocess_and_save_or_load_sequences(self, filename, dataset, labels):
    if os.path.exists(filename):
      return self.load_file(filename)
    else:
      data = {}
      sequences, sequence_labels = self.make_sequences(dataset, labels)
      data['sequences'] = sequences
      data['labels'] = sequence_labels
      with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return data

  # images should already be shuffled.
  def make_sequences(self, images, image_labels, max_length=5):
    n_sequenced = 0
    sequences = []
    labels = []
    while len(images) > 0:
      images, image_labels, sequence, sequence_label = self.make_sequence(images, image_labels, max_length, delete=True)
      sequences.append(sequence)
      labels.append(sequence_label)
      if len(sequences) % 500 == 0:
        print("...")
        print(len(sequences))
        print(n_sequenced)
        print(len(images))
      n_sequenced += int(sequence_label[0])

    return sequences, labels

  def make_sequence(self, images, image_labels, max_length, delete=False):
    length = min(len(images), int(np.floor(max_length * np.random.rand()) + 1))
    sequence_images = []
    digit_labels = np.zeros((5, 11))
    for i in range(length):
      index = np.random.randint(0, len(images))
      sequence_images.append(images[index])

      digit_labels[i, image_labels[index]] = 1
      if delete:
        image_labels = np.delete(image_labels, index, axis=0)
        images = np.delete(images, index, axis=0)
    for i in range(max_length - length):
      digit_labels[i+length, 10] = 1
    sequence, bbox = self.apply_random_transforms_sequence(sequence_images)
    sequence_label = np.concatenate((
      [length],
      bbox,
      digit_labels.reshape(55)
    ))
    return images, image_labels, sequence, sequence_label

  def apply_random_transforms_sequence(self, images):
    if np.random.rand() < 0.7:
      sequence = self.set_random_axis(images)
    else:
      sequence = self.set_horizontal_axis(images)

    sequence, bbox = self.wrap_sequence(sequence)

    n_transforms = 3.
    rand_transform = np.random.rand()

    if rand_transform < 1/n_transforms:
      sequence = self.rotate(sequence)

    elif rand_transform < 2/n_transforms:
      sequence = self.translate(sequence)

    else:
      sequence = self.scale(sequence)

    return sequence, bbox

  #
  # Random single image transformations
  #

  def rotate(self, img, max_theta=15.):
    theta = (np.random.rand() - 0.5)*2 * max_theta
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    return cv2.warpAffine(img,M,(cols,rows))

  def translate(self, image, max_translation=3):
    x = np.ceil((np.random.rand()-0.5)*2 * max_translation)
    y = np.ceil((np.random.rand()-0.5)*2 * max_translation)
    rows,cols = image.shape
    M = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image,M,(cols,rows))

  def scale(self, img, max_factor=0.07, interpolation=cv2.INTER_LINEAR):
    factor = 1. - 2*(np.random.rand()-0.5) * max_factor
    original_shape = img.shape
    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation = interpolation)
    return self.crop_or_pad(img, shape=original_shape)

  def affine_transform(self, img, max_radius=3):
    rows,cols = img.shape
    max_translate = np.sqrt(max_radius)
    # select 3 random points
    pts = np.floor(np.random.rand(3, 2) * 29)

    pts2 = np.zeros((3, 2))
    for i in range(len(pts)):
      # move each of those points by max of max_translation
      x = np.max((0., np.min((28., (pts[i][0] + np.ceil(2*(np.random.rand()-0.5) * max_translate))))))
      y = np.max((0., np.min((28., (pts[i][1] + np.ceil(2*(np.random.rand()-0.5) * max_translate))))))
      pts2[i][0] = x
      pts2[i][1] = y

    M = cv2.getAffineTransform(np.float32(pts),np.float32(pts2))

    return cv2.warpAffine(img,M,(cols,rows))

  def perspective_transform(self, img, border=3, max_stretch=2):
    rows,cols = img.shape
    pts = np.array([
      [np.random.randint(0, border+1), np.random.randint(0, border+1)],
      [np.random.randint(0, border+1), np.random.randint(rows-border, rows+1)],
      [np.random.randint(cols-border, cols+1), np.random.randint(0, border+1)],
      [np.random.randint(cols-border, cols+1), np.random.randint(rows-border, rows+1)]
    ])

    pts2 = np.zeros((4, 2))
    for i in range(len(pts)):
      x = np.max((0., np.min((28., (pts[i][0] + np.ceil(2*(np.random.rand()-0.5) * max_stretch))))))
      y = np.max((0., np.min((28., (pts[i][1] + np.ceil(2*(np.random.rand()-0.5) * max_stretch))))))
      pts2[i][0] = x
      pts2[i][1] = y

    M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(pts2))

    return cv2.warpPerspective(img,M,(28,28))

  def crop_or_pad(self, img, padding=(0,0), shape=(28,28)):
    cx = np.floor(shape[1]/2.)
    width = shape[1]
    padx = padding[1]

    displacex = cx + padx - np.floor(img.shape[1]/2.)
    if displacex < 0:
      img = img[:, abs(displacex):]
    elif displacex > 0:
      add_zeros = np.zeros((img.shape[0], displacex + img.shape[1]))
      add_zeros[:, displacex:] = img
      img = add_zeros

    overflowx = img.shape[1] - width
    if overflowx > 0:
      img = img[:, :-overflowx]
    elif overflowx < 0:
      add_zeros = np.zeros((img.shape[0], img.shape[1] + abs(overflowx)))
      add_zeros[:, :img.shape[1]] = img
      img = add_zeros

    cy = np.floor(shape[0]/2.)
    height = shape[0]
    pady = padding[0]

    displacey = cy + pady - np.floor(img.shape[0]/2.)
    if displacey < 0:
      img = img[abs(displacey):, :]
    elif displacey > 0:
      add_zeros = np.zeros((displacey + img.shape[0], img.shape[1]))
      add_zeros[displacey:, :] = img
      img = add_zeros

    overflowy = img.shape[0] - height
    if overflowy > 0:
      img = img[:-overflowy, :]
    elif overflowy < 0:
      add_zeros = np.zeros((img.shape[0] + abs(overflowy), img.shape[1]))
      add_zeros[:img.shape[0], :] = img
      img = add_zeros

    return img

  def wrap_sequence(self, sequence, shape=(28*5*2, 28*5*2), permitted_hanging=1):
    top = np.round(np.random.rand() * (shape[0] + permitted_hanging - sequence.shape[0]))
    left = np.round(np.random.rand() * (shape[1] + permitted_hanging - sequence.shape[1]))

    overflowy = (top + sequence.shape[0]) - shape[0]
    if overflowy > 0:
      sequence = sequence[:(sequence.shape[0] - overflowy), :]
    overflowx = (left + sequence.shape[1]) - shape[1]
    if overflowx > 0:
      sequence = sequence[:, :(sequence.shape[1] - overflowx)]
    canvas = np.zeros(shape)

    canvas[top:(top+sequence.shape[0]), left:(left+sequence.shape[1])] = sequence
    return canvas, np.array([left, top, sequence.shape[1], sequence.shape[0]])

  def set_horizontal_axis(self, images):
    max_height = max(image.shape[0] for image in images)
    total_width = sum(image.shape[1] for image in images)
    sequence = np.zeros((max_height, total_width))
    left = 0;
    for image in images:
      sequence[0:image.shape[0], left:(left+image.shape[1])] = image
      left += image.shape[1]
    return sequence

  def set_random_axis(self, images):
    offset = np.floor(np.random.rand() * 29)
    total_offset = (len(images)-1) * offset
    total_height = sum(image.shape[0] for image in images)
    total_width = sum(image.shape[1] for image in images)

    if np.random.rand() < 0.1:
      # set diagonally ascending.
      max_height = 0
      for i in range(len(images)):
        max_height = np.max((max_height, images[i].shape[0] + (total_offset - i*offset)))

      top = total_offset
      left = 0
      sequence = np.zeros((max_height, total_width))
      for image in images:
        height = image.shape[0]
        width = image.shape[1]
        sequence[top:(top+height), left:(left+width)] = image
        top -= offset
        left += width
      return sequence

    elif np.random.rand() < 0.55:
      # set diagonally descending, offset vertically
      max_height = 0
      for i in range(len(images)):
        max_height = np.max((max_height, images[i].shape[0] + i*offset))
      top = 0
      left = 0
      sequence = np.zeros((max_height, total_width))
      for image in images:
        height = image.shape[0]
        width = image.shape[1]
        sequence[top:(top+height), left:(left+width)] = image
        top += offset
        left += width
      return sequence
    else:
      # set diagonally descending, offset horizontally
      max_width = 0
      for i in range(len(images)):
        max_width = np.max((max_width, images[i].shape[1] + i*offset))
      top = 0
      left = 0
      sequence = np.zeros((total_height, max_width))
      for image in images:
        height = image.shape[0]
        width = image.shape[1]
        sequence[top:(top+height), left:(left+width)] = image
        top += height
        left += offset
      return sequence
