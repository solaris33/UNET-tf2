import tensorflow as tf

# UNET for ISBI-2012 dataset
class UNET_ISBI_2012(tf.keras.Model):
  def __init__(self, num_classes):
    super(UNET_ISBI_2012, self).__init__()

    # Input
    inputs = tf.keras.layers.Input((512,512,1))
    
    # Contracting part
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    assert conv1.shape[1:] == (512, 512, 64)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    assert pool1.shape[1:] == (256, 256, 64)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    assert conv2.shape[1:] == (256, 256, 128)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    assert pool2.shape[1:] == (128, 128, 128)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    assert conv3.shape[1:] == (128, 128, 256)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    assert pool3.shape[1:] == (64, 64, 256)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    assert drop4.shape[1:] == (64, 64, 512)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    assert pool4.shape[1:] == (32, 32, 512)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)    
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    assert conv5.shape[1:] == (32, 32, 1024)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Expansive part
    up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    assert up6.shape[1:] == (64, 64, 512)
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    assert merge6.shape[1:] == (64, 64, 1024)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    assert conv6.shape[1:] == (64, 64, 512)

    up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    assert up7.shape[1:] == (128, 128, 256)
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    assert merge7.shape[1:] == (128, 128, 512)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    assert conv7.shape[1:] == (128, 128, 256)

    up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    assert up8.shape[1:] == (256, 256, 128)
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    assert merge8.shape[1:] == (256, 256, 256)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    assert conv8.shape[1:] == (256, 256, 128)

    up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    assert up9.shape[1:] == (512, 512, 64)
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    assert merge9.shape[1:] == (512, 512, 128)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    assert conv9.shape[1:] == (512, 512, 64)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    assert conv9.shape[1:] == (512, 512, 2)
    conv10 = tf.keras.layers.Conv2D(num_classes, 1, activation = 'sigmoid')(conv9)
    assert conv10.shape[1:] == (512, 512, num_classes)

    model = tf.keras.Model(inputs = inputs, outputs = conv10)

    self.model = model

    # print model structure
    self.model.summary()

  def call(self, x):
    return self.model(x)

# UNET for Oxford-IIIT dataset
class UNET_OXFORD_IIIT(tf.keras.Model):
  def __init__(self, num_classes):
    super(UNET_OXFORD_IIIT, self).__init__()

    # Input
    inputs = tf.keras.layers.Input((128,128,3))
    
    # Contracting part
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    assert conv1.shape[1:] == (128, 128, 64)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    assert pool1.shape[1:] == (64, 64, 64)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    assert conv2.shape[1:] == (64, 64, 128)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    assert pool2.shape[1:] == (32, 32, 128)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    assert conv3.shape[1:] == (32, 32, 256)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    assert pool3.shape[1:] == (16, 16, 256)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    assert drop4.shape[1:] == (16, 16, 512)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    assert pool4.shape[1:] == (8, 8, 512)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)    
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    assert conv5.shape[1:] == (8, 8, 1024)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Expansive part
    up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    assert up6.shape[1:] == (16, 16, 512)
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    assert merge6.shape[1:] == (16, 16, 1024)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    assert conv6.shape[1:] == (16, 16, 512)

    up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    assert up7.shape[1:] == (32, 32, 256)
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    assert merge7.shape[1:] == (32, 32, 512)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    assert conv7.shape[1:] == (32, 32, 256)

    up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    assert up8.shape[1:] == (64, 64, 128)
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    assert merge8.shape[1:] == (64, 64, 256)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    assert conv8.shape[1:] == (64, 64, 128)

    up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    assert up9.shape[1:] == (128, 128, 64)
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    assert merge9.shape[1:] == (128, 128, 128)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    assert conv9.shape[1:] == (128, 128, 64)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    assert conv9.shape[1:] == (128, 128, 2)
    conv10 = tf.keras.layers.Conv2D(num_classes, 1, activation = None)(conv9)
    assert conv10.shape[1:] == (128, 128, num_classes)

    model = tf.keras.Model(inputs = inputs, outputs = conv10)

    self.model = model

    # print model structure
    self.model.summary()

  def call(self, x):
    return self.model(x)