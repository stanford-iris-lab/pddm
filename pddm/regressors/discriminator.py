import tensorflow as tf


def discriminator_network(
    inputStates, inputSize, num_fc_layers, depth_fc_layers, tf_datatype, scope
):

    outputSize = 1  # disc has 1 output per datapoint

    with tf.variable_scope(str(scope)):

        # concat K entries together [bs x K x sa] --> [bs x ksa]
        inputState =  tf.layers.flatten(inputStates)

        # vars
        intermediate_size = depth_fc_layers
        reuse = False
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=None, dtype=tf_datatype
        )
        fc = tf.contrib.layers.fully_connected

        # make hidden layers
        for i in range(num_fc_layers):
            if i == 0:
                fc_i = fc(
                    inputState,
                    num_outputs=intermediate_size,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=initializer,
                    reuse=reuse,
                    trainable=True,
                )
            else:
                fc_i = fc(
                    h_i,
                    num_outputs=intermediate_size,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=initializer,
                    reuse=reuse,
                    trainable=True,
                )
            h_i = tf.nn.relu(fc_i)

        # make output layer
        z = fc(
            h_i,
            num_outputs=outputSize,
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=initializer,
            reuse=reuse,
            trainable=True,
        )

    return z
