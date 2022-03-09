import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class CNN:
    @staticmethod
    def model(**args):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(65, 41, 1), padding='same'),
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
                tf.keras.layers.Dense(50, activation='softmax')
        ])
        return model

    class callback(tf.keras.callbacks.Callback):
        def __init__(self, callrate=.99):
            super().__init__()
            self.callrate = callrate
        def on_epoch_end(self, epoch, los={}):
            if los['val_accuracy'] >= self.callrate:
                self.model.stop_training = True
            