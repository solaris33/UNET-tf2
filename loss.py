import tensorflow as tf

# binary cross entropy for ISBI-2012 dataset
binary_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# sparse categorical cross etnropy for Oxford-IIIT dataset
sparse_categorical_cross_entropy_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)