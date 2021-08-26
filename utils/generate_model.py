import tensorflow as tf

# FEMNIST image size
IMAGE_SIZE = 28
# FEMNIST number of classes
NUM_CLASSES = 62

# CNN model generator for FEMNIST using sequential
def generate_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1), input_shape=(IMAGE_SIZE * IMAGE_SIZE,)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
    model.add(tf.keras.layers.Dense(units=NUM_CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    return model