import tensorflow as tf
import tensorflow_datasets as tfds

from absl import flags
from absl import app

import matplotlib.pyplot as plt

import os

from model import UNET_OXFORD_IIIT

flags.DEFINE_string('checkpoint_path', default='saved_model_oxford_iiit/unet_model.h5', help='path to a directory to restore checkpoint file')
flags.DEFINE_string('test_dir', default='oxford_iiit_test_result', help='directory which test prediction result saved')
flags.DEFINE_integer('num_classes', default=3, help='number of prediction classes')

FLAGS = flags.FLAGS

# set configuration value
batch_size = 1

# load pascal oxford-iit dataset using tfds
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def normalize_oxford_iiit(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1  # set value 1,2,3 -> 0,1,2
  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize_oxford_iiit(input_image, input_mask)

  return input_image, input_mask

test = dataset['test'].map(load_image_test)
test_data = test.batch(batch_size)

# make prediction mask
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]

  return pred_mask[0]

# display image and save
def display_and_save(display_list, save_path):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.savefig(save_path)

def main(_):
  # check if checkpoint path exists
  if not os.path.exists(FLAGS.checkpoint_path):
    print('checkpoint file is not exists!')
    exit()

  # create UNET model
  unet_model = UNET_OXFORD_IIIT(FLAGS.num_classes)

  # restore latest checkpoint
  unet_model.load_weights(FLAGS.checkpoint_path)
  print(f'{FLAGS.checkpoint_path} checkpoint is restored!')

  # check total image num
  print('total test image :', info.splits['test'])

  # save test prediction result to png file
  if not os.path.exists(os.path.join(os.getcwd(), FLAGS.test_dir)):
    os.mkdir(os.path.join(os.getcwd(), FLAGS.test_dir))

  for image_num, test_batch in enumerate(test_data):
    test_image, test_mask = test_batch
    test_image = test_image[0]
    test_mask = test_mask[0]

    output_image_path = os.path.join(os.getcwd(), FLAGS.test_dir, f'{image_num}_result.png')
    display_and_save([test_image, test_mask, create_mask(unet_model.predict(test_image[tf.newaxis, ...]))], output_image_path)
    print(output_image_path + ' saved!')

if __name__ == '__main__':
  app.run(main)