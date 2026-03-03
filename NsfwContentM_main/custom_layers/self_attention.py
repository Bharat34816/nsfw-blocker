import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time, features)
        self.context = self.add_weight(
            name="context_vector",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        scores = tf.tensordot(inputs, self.context, axes=1)   # (batch, time)
        weights = tf.nn.softmax(scores, axis=1)               # (batch, time)
        output = tf.reduce_sum(
            inputs * tf.expand_dims(weights, -1),
            axis=1
        )  # (batch, features)
        return output

    def get_config(self):
        return super().get_config()
