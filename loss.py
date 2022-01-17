import tensorflow as tf

# binary cross entropy for ISBI-2012 dataset
binary_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)