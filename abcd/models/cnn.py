import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class CNN:
    @staticmethod
    def model(height, width, depth, classes, **args):

        inputShape = (height, width, depth)
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=inputShape, padding='same'),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(classes, activation='softmax')
        ])
        return model

    class callback(tf.keras.callbacks.Callback):
        def __init__(self, val_rate= .99):
            super().__init__()
            self.val_rate = val_rate
        def on_epoch_end(self, epoch, los={}):
            if los['val_accuracy'] >= 0.99:
                self.model.stop_training = True
            