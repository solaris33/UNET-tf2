import tensorflow as tf
import tensorflow_datasets as tfds

from absl import flags
from absl import app

import os
import matplotlib.pyplot as plt
import numpy as np

from model import UNET_OXFORD_IIIT
from loss import sparse_categorical_cross_entropy_object

# set seed
tf.random.set_seed(1234)

flags.DEFINE_string('checkpoint_path', default='saved_model_oxford_iiit/unet_model.h5', help='path to a directory to save model checkpoints during training')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log_oxford_iiit', help='path to a directory to save tensorboard log')
flags.DEFINE_integer('num_epochs', default=5, help='training epochs')
flags.DEFINE_integer('steps_per_epoch', default=2000, help='steps per epoch')
flags.DEFINE_integer('num_classes', default=3, help='number of prediction classes')

FLAGS = flags.FLAGS

# set configuration value
batch_size = 64
learning_rate = 0.001
buffer_size = 1000

# load pascal oxford-iit dataset using tfds
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def normalize_oxford_iiit(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1  # set value 1,2,3 -> 0,1,2
  return input_image, input_mask

def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize_oxford_iiit(input_image, input_mask)

  return input_image, input_mask

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train.cache().shuffle(buffer_size).batch(batch_size).repeat()
train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# display image
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# display image and save
def display_and_save(display_list, epoch):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.savefig(f'epoch {epoch}.jpg')

# make prediction mask
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]

  return pred_mask[0]

# show prediction
def show_predictions(model, sample_image, sample_mask):
  display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

# display and save prediction
def save_predictions(epoch, model, sample_image, sample_mask):
  display_and_save([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))], epoch)

# set custom callback
class CustomCallback(tf.keras.callbacks.Callback):
  def __init__(self, unet_model, sample_image, sample_mask):
    super(CustomCallback, self).__init__()
    self.unet_model = unet_model
    self.sample_image = sample_image
    self.sample_mask = sample_mask

  def on_epoch_end(self, epoch, logs=None):
    save_predictions(epoch+1, self.unet_model, self.sample_image, self.sample_mask)
    print (f'\n에포크 이후 예측 예시 {epoch+1}\n')

def main(_):
  # data sanity check
  for iter, batch_data in enumerate(train_data):
    if iter >= 2:  # manually detect the end of the epoch
      break
    batch_image, batch_mask = batch_data[0], batch_data[1]
    sample_image, sample_mask = batch_image[0], batch_mask[0]

  # data display
  display([sample_image, sample_mask])

  # create ISBI-2012 UNET model
  unet_model = UNET_OXFORD_IIIT(FLAGS.num_classes)

  # show prediction before training
  show_predictions(unet_model, sample_image, sample_mask)

  # set optimizer
  optimizer = tf.optimizers.Adam(learning_rate) 

  # check if checkpoint path exists
  if not os.path.exists(FLAGS.checkpoint_path.split('/')[0]):
    os.mkdir(FLAGS.checkpoint_path.split('/')[0])

  # restore latest checkpoint
  if os.path.isfile(FLAGS.checkpoint_path):
    unet_model.load_weights(FLAGS.checkpoint_path)
    print(f'{FLAGS.checkpoint_path} checkpoint is restored!')

  # set callback
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(FLAGS.checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_log_path)
  custom_callback = CustomCallback(unet_model, sample_image, sample_mask)

  # set compile
  unet_model.compile(optimizer = optimizer, loss = sparse_categorical_cross_entropy_object, metrics = ['accuracy'])

  # start training
  unet_model.fit_generator(train_data,
                          steps_per_epoch=FLAGS.steps_per_epoch,
                          epochs=FLAGS.num_epochs,
                          callbacks=[model_checkpoint_callback, tensorboard_callback, custom_callback])

if __name__ == '__main__':
  app.run(main)