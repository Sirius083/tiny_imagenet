"""
Tiny ImageNet: Input Pipeline
Written by Patrick Coady (pcoady@alum.mit.edu)

Reads in jpegs, distorts images (flips, translations, hue and
saturation) and builds QueueRunners to keep the GPU well-fed. Uses
specific directory and file naming structure from data download
link below.

Also builds dictionary between label integer and human-readable
class names.

Get data here:
https://tiny-imagenet.herokuapp.com/
"""
import glob
import re
import tensorflow as tf
import random
import numpy as np


def load_filenames_labels(mode):
  """Gets filenames and labels

  Args:
    mode: 'train' or 'val'
      (Directory structure and file naming different for
      train and val datasets)

  Returns:
    list of tuples: (jpeg filename with path, label)
  """
  label_dict, class_description = build_label_dicts()
  filenames_labels = []
  if mode == 'train':
    # sirius: glob.glob: Return a list of paths matching a pathname pattern.
    # sirius: 更改过将train下面的iamges文件夹去掉了
    filenames = glob.glob('E:/tiny_imagenet/tiny-imagenet-200/train/*/*.JPEG')
    for filename in filenames:
      match = re.search(r'n\d+', filename)
      label = str(label_dict[match.group()])
      filenames_labels.append((filename, label))
  elif mode == 'val':
    with open('E:/tiny_imagenet/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = 'E:/tiny_imagenet/tiny-imagenet-200/val/images/' + split_line[0]
        label = str(label_dict[split_line[1]])
        filenames_labels.append((filename, label))

  return filenames_labels


def build_label_dicts():
  """Build look-up dictionaries for class label, and class description

  Class labels are 0 to 199 in the same order as 
    tiny-imagenet-200/wnids.txt. 
  Class text descriptions are from 
    tiny-imagenet-200/words.txt

  Returns:
    tuple of dicts
      label_dict: 
        keys = synset (e.g. "n01944390")
        values = class integer {0 .. 199}
      class_desc:
        keys = class integer {0 .. 199}
        values = text description from words.txt
  """
  label_dict, class_description = {}, {}
  with open('E:/tiny_imagenet/tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i
  with open('E:/tiny_imagenet/tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t') # 同义词
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc

  return label_dict, class_description


def read_image(filename_q, mode):
  """Load next jpeg file from filename / label queue
  Randomly applies distortions if mode == 'train' (including a 
  random crop to [56, 56, 3]). Standardizes all images.

  Args:
    filename_q: Queue with 2 columns: filename string and label string.
     filename string is relative path to jpeg file. label string is text-
     formatted integer between '0' and '199'
    mode: 'train' or 'val'

  Returns:
    [img, label]: 
      img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
      label = tf.unit8 target class label: {0 .. 199}
  """
  item = filename_q.dequeue()
  filename = item[0]
  label = item[1]
  file = tf.read_file(filename)
  img = tf.image.decode_jpeg(file, channels=3)
  # image distortions: left/right, random hue(色调), random color saturation
  if mode == 'train':
    img = tf.random_crop(img, np.array([56, 56, 3]))
    img = tf.image.random_flip_left_right(img)
    # val accuracy improved without random hue
    # img = tf.image.random_hue(img, 0.05)
    img = tf.image.random_saturation(img, 0.5, 2.0) # 色彩饱和度
  else:
    img = tf.image.crop_to_bounding_box(img, 4, 4, 56, 56) # offset_height, offset_width, target_height, target_width

  label = tf.string_to_number(label, tf.int32)
  label = tf.cast(label, tf.uint8)

  return [img, label]


def batch_q(mode, config):
  """Return batch of images using filename Queue

  Args:
    mode: 'train' or 'val'
    config: training configuration object

  Returns:
    imgs: tf.uint8 tensor [batch_size, height, width, channels]
    labels: tf.uint8 tensor [batch_size,]

  """
  filenames_labels = load_filenames_labels(mode)
  # print('filenames_labels', filenames_labels) []
  random.shuffle(filenames_labels)
  filename_q = tf.train.input_producer(filenames_labels,
                                       num_epochs=config.num_epochs,
                                       shuffle=True) # FIFOQueue object

  # 2 read_image threads to keep batch_join queue full:
  # batch_join: Runs a list of tensors to fill a queue to create batches of examples.
  return tf.train.batch_join([read_image(filename_q, mode) for i in range(2)],
                             config.batch_size, shapes=[(56, 56, 3), ()],
                             capacity=2048)



