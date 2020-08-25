import tensorflow as tf
import os
from pathlib import Path

class Model(tf.keras.Model):
    def __init__(self, padding):
        super().__init__()
        self.up1 = tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=(5, 3),
                strides=(2, 1),
                padding=padding,
                data_format='channels_last',
                dilation_rate=(1, 1),
                use_bias=True,
                kernel_initializer='he_normal',
                bias_initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        return self.up1(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

def add_inputs(model, shape):
    input = tf.random.uniform(shape)
    output = model(input)

    print("tests      ", output.shape)

    return model

def store_model_as_tfl(base_model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as file:
        file.write(tflite_model)

def save_model(name, model, shape):
    add_inputs(model, shape)

    store_model_as_tfl(model, '../app/src/main/assets/' + name)
    print('Done model', name)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    save_model("model_01.tflite", Model(padding='same'), (1, 9, 6, 45))
    save_model("model_02.tflite", Model(padding='valid'), (1, 9, 6, 45))

    save_model("model_03.tflite", Model(padding='same'), (1, 10, 6, 45))
    save_model("model_04.tflite", Model(padding='valid'), (1, 10, 6, 45)) 

