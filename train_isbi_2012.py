import tensorflow as tf

from absl import flags
from absl import app

import os
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from model import UNET_ISBI_2012
from loss import binary_loss_object

# set seed
tf.random.set_seed(1234)

flags.DEFINE_string('checkpoint_path', default='saved_model_isbi_2012/unet_model.h5', help='path to a directory to save model checkpoints during training')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log_isbi_2012', help='path to a directory to save tensorboard log')
flags.DEFINE_integer('num_epochs', default=5, help='training epochs')
flags.DEFINE_integer('steps_per_epoch', default=2000, help='steps per epoch')
flags.DEFINE_integer('num_classes', default=1, help='number of prediction classes')

FLAGS = flags.FLAGS

# set configuration value
batch_size = 2
learning_rate = 0.0001

# normalize isbi-2012 data
def normalize_isbi_2012(input_images, mask_labels):
  # 0~255 -> 0.0~1.0
  input_images = input_images / 255
  mask_labels = mask_labels / 255

  # set label to binary
  mask_labels[mask_labels > 0.5] = 1
  mask_labels[mask_labels <= 0.5] = 0

  return input_images, mask_labels

# make data generator
def make_train_generator(batch_size, aug_dict):
  image_gen = ImageDataGenerator(**aug_dict)
  mask_gen = ImageDataGenerator(**aug_dict)

  # set image and mask same augmentation using same seed 
  image_generator = image_gen.flow_from_directory(
      directory='./isbi_2012/preprocessed',
      classes = ['train_imgs'],
      class_mode = None,
      target_size = (512, 512),
      batch_size = batch_size,
      color_mode='grayscale',
      seed=1
      )
  mask_generator = mask_gen.flow_from_directory(
      directory='./isbi_2012/preprocessed',
      classes = ['train_labels'],
      class_mode = None,
      target_size = (512, 512),
      batch_size = batch_size,
      color_mode='grayscale',
      seed=1
      )
  train_generator = zip(image_generator, mask_generator)
  for (batch_images, batch_labels) in train_generator:
    batch_images, batch_labels = normalize_isbi_2012(batch_images, batch_labels)
    
    yield (batch_images, batch_labels)

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
  pred_mask = np.where(pred_mask > 0.5, 1, 0)

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
  # set augmentation
  aug_dict = dict(rotation_range=0.2,
                      width_shift_range=0.05,
                      height_shift_range=0.05,
                      shear_range=0.05,
                      zoom_range=0.05,
                      horizontal_flip=True,
                      fill_mode='nearest')

  # make generator
  train_generator = make_train_generator(batch_size, aug_dict)

  # data sanity check
  for iter, batch_data in enumerate(train_generator):
    if iter >= 2:  # manually detect the end of the epoch
      break
    batch_image, batch_mask = batch_data[0], batch_data[1]
    sample_image, sample_mask = batch_image[0], batch_mask[0]

  # data display
  display([sample_image, sample_mask])

  # create ISBI-2012 UNET model
  unet_model = UNET_ISBI_2012(FLAGS.num_classes)

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
  unet_model.compile(optimizer = optimizer, loss = binary_loss_object, metrics = ['accuracy'])

  # start training
  unet_model.fit_generator(train_generator,
                           steps_per_epoch=FLAGS.steps_per_epoch,
                           epochs=FLAGS.num_epochs,
                           callbacks=[model_checkpoint_callback, tensorboard_callback, custom_callback])

if __name__ == '__main__':
  app.run(main)