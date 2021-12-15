import tensorflow as tf
from tensorflow.keras.regularizers import l2

# ICHAR number of classes
NUM_CLASSES = 9

# ICHAR channels
NUM_CHANNELS = 6

# CNN model generator for ICHAR using sequential
def generate_model_ichar():
    regularizer_factor = 0.001
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, activation='relu', input_shape=(256, NUM_CHANNELS),
        #kernel_regularizer=tf.keras.regularizers.l2(regularizer_factor), bias_regularizer=tf.keras.regularizers.l2(regularizer_factor)
    )) # 254 x 32
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2)) # 127 x 32
    model.add(tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, activation='relu',
        #kernel_regularizer=tf.keras.regularizers.l2(regularizer_factor), bias_regularizer=tf.keras.regularizers.l2(regularizer_factor)
    )) # 125 x 64
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2)) # 62 x 64
    model.add(tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, activation='relu',
        #kernel_regularizer=tf.keras.regularizers.l2(regularizer_factor), bias_regularizer=tf.keras.regularizers.l2(regularizer_factor)
    )) # 60 x 128
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2)) # 30 x 128
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=256, activation='relu',
        #kernel_regularizer=tf.keras.regularizers.l2(regularizer_factor), bias_regularizer=tf.keras.regularizers.l2(regularizer_factor)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=NUM_CLASSES,
        #kernel_regularizer=tf.keras.regularizers.l2(regularizer_factor), bias_regularizer=tf.keras.regularizers.l2(regularizer_factor)
    ))
    return model