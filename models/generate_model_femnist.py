import tensorflow as tf
from tensorflow.keras.regularizers import l2

# FEMNIST image size
IMAGE_SIZE = 28
# FEMNIST number of classes
NUM_CLASSES = 62

# CNN model generator for FEMNIST using sequential
def generate_model_femnist():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1), input_shape=(IMAGE_SIZE * IMAGE_SIZE,)))
    model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=(5, 5), padding='same', activation='relu',
    ))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(5, 5), padding='same', activation='relu',
    ))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=2048, activation='relu',
    ))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=NUM_CLASSES,
    ))
    return model