import tensorflow as tf


def attention_model(inputs, attention_size):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value

    weights = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    biases = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    attention_weights = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('layer_op'):
        layer_op = tf.tanh(tf.tensordot(inputs, weights, axes=1) + biases)

    dot_prod = tf.tensordot(layer_op, attention_weights, axes=1, name='dot_prod')
    alphas = tf.nn.softmax(dot_prod, name='alphas')

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output