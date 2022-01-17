# Preprocessing isbi 2012 dataset tiff to png
import tifffile as tiff
import skimage.io as io
import os

preprocessed_train_img_folder_path = os.path.join('isbi_2012', 'preprocessed', 'train_imgs')
preprocessed_train_label_folder_path = os.path.join('isbi_2012', 'preprocessed', 'train_labels')
preprocessed_test_img_folder_path = os.path.join('isbi_2012', 'preprocessed', 'test_imgs')

train_images = tiff.imread(os.path.join('isbi_2012', 'raw_data', 'train-volume.tif'))
train_labels = tiff.imread(os.path.join('isbi_2012', 'raw_data', 'train-labels.tif'))
test_images = tiff.imread(os.path.join('isbi_2012', 'raw_data', 'test-volume.tif'))

print('train img tiff file shape :', train_images.shape)
print('train label tiff file shape :',train_labels.shape)
print('test img tiff file shape :', test_images.shape)

# check if preprocessing folder path exists
if not os.path.exists(preprocessed_train_img_folder_path):
  os.mkdir(preprocessed_train_img_folder_path)
  os.mkdir(preprocessed_train_label_folder_path)
  os.mkdir(preprocessed_test_img_folder_path)

for image_index, zip_element in enumerate(zip(train_images, train_labels, test_images)):
  each_train_image, each_train_label, each_test_image = zip_element

  io.imsave(os.path.join(preprocessed_train_img_folder_path, f"{image_index}.png"), each_train_image)
  io.imsave(os.path.join(preprocessed_train_label_folder_path, f"{image_index}.png"), each_train_label)
  io.imsave(os.path.join(preprocessed_test_img_folder_path, f"{image_index}.png"), each_test_image)

print('ISBI 2012 Preprocessing finished!')