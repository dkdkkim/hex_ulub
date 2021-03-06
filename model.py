__author__ = 'Haohan Wang'

import tensorflow as tf

def lamda_variable(shape): # NGLCM related
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("lamda", shape, initializer=initializer, dtype=tf.float32)


def theta_variable(shape): # NGLCM related
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("theta", shape, initializer=initializer, dtype=tf.float32)


def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape, initializer=initializer, dtype=tf.float32)


def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels / groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


class AlexNet(object): # Examples of original AlexNet for ImageNet classification
    def __init__(self, x, y):
        self.x = tf.reshape(x, shape=[-1, 227, 227, 3])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.top_k = tf.placeholder(tf.int64)
        self.e = tf.placeholder(tf.float32)
        self.batch = tf.placeholder(tf.float32)
        self.WEIGHTS_PATH = 'weights/bvlc_alexnet.npy'
        self.NUM_CLASSES = 2

        # conv1
        conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        self.rep = fc(dropout6, 4096, 4096, name='fc7')

        dropout7 = dropout(self.rep, self.keep_prob)

        # 8th Layer: FC and return unscaled activations
        y_conv_loss = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

        # fc2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_loss, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv_loss, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        topk_correct = tf.nn.in_top_k(y_conv_loss, tf.argmax(y, 1), k=self.top_k)
        self.topk_accuracy = tf.reduce_mean(tf.cast(topk_correct, tf.float32))

   


class AlexNetHex(object): # Examples of plugging in HEX and NGLCM into original AlexNet for ImageNet classification
    def __init__(self, x, y, x_re, x_d, conf, Hex_flag=False):
        self.x = tf.reshape(x, shape=[-1, 227, 227, 3])
        self.x_re = tf.reshape(x_re, shape=[-1, 1, 128*128])
        self.x_d = tf.reshape(x_d, shape=[-1, 1, 128*128])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.top_k = tf.placeholder(tf.int64)
        self.e = tf.placeholder(tf.float32)
        self.batch = tf.placeholder(tf.float32)
        self.WEIGHTS_PATH = 'weights/bvlc_alexnet.npy'

        # NGLCM
        with tf.variable_scope('nglcm'):
            lamda = lamda_variable([conf.ngray, 1])
            theta = theta_variable([conf.ngray, 1])
            g = tf.matmul(tf.minimum(tf.maximum(tf.subtract(self.x_d, lamda), 1e-5), 1),
                          tf.minimum(tf.maximum(tf.subtract(self.x_re, theta), 1e-5), 1), transpose_b=True)

        with tf.variable_scope("nglcm_fc1"):
            g_flat = tf.reshape(g, [-1, conf.ngray * conf.ngray])
            glgcm_W_fc1 = weight_variable([conf.ngray * conf.ngray, 32])
            glgcm_b_fc1 = bias_variable([32])
            glgcm_h_fc1 = tf.nn.relu(tf.matmul(g_flat, glgcm_W_fc1) + glgcm_b_fc1)
        # --------------------------

        # 1st Layer
        conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')

        fc7 = tf.nn.l2_normalize(fc7, 0)

        h_fc_drop = dropout(fc7, self.keep_prob)

        # Empirically, we noticed that normalization helps improving the performance, it is recommended, but not necessary
        glgcm_h_fc1 = tf.nn.l2_normalize(glgcm_h_fc1, 0)
        h_fc_drop = tf.nn.l2_normalize(h_fc_drop, 0)
        # --------------------------

        # concatenate the representations (Equation 3 in paper)
        yconv_contact_loss = tf.concat([h_fc_drop, glgcm_h_fc1], 1)
        pad = tf.zeros_like(glgcm_h_fc1, tf.float32)
        yconv_contact_pred = tf.concat([h_fc_drop, pad], 1)
        pad2 = tf.zeros_like(fc7, tf.float32)
        yconv_contact_H = tf.concat([pad2, glgcm_h_fc1], 1)
        # --------------------------

        # Expanding the final layer to take in the representation summarized by NGLCM
        with tf.variable_scope("fc8"):
#            W_fc2 = weight_variable([4128, 1000])
#            b_fc2 = bias_variable([1000])
            W_fc2 = weight_variable([4128, 2])
            b_fc2 = bias_variable([2])
            y_conv_loss = tf.matmul(yconv_contact_loss, W_fc2) + b_fc2
            y_conv_pred = tf.matmul(yconv_contact_pred, W_fc2) + b_fc2
            y_conv_H = tf.matmul(yconv_contact_H, W_fc2) + b_fc2
        # --------------------------

        # Different roles of y_conv_pred and y_conv_loss, for the purpose of succinctness
        # One can also replace the following y_conv_pred to y_conv_loss, empirically, we do not observe big differences
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_pred, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        topk_correct = tf.nn.in_top_k(y_conv_pred, tf.argmax(y, 1), k=self.top_k)
        self.topk_accuracy = tf.reduce_mean(tf.cast(topk_correct, tf.float32))
        # --------------------------

        if Hex_flag:
            # Projection (Equation 4 in the paper)
            # Notice that, we are using the most succinct form of HEX as an example
            y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(tf.matmul(y_conv_H, y_conv_H, transpose_a=True))),
                              y_conv_H, transpose_b=True), y_conv_loss)
            # --------------------------


            # # another form that involves a hyperparameter which can help the superficial statistics learner to summarize related statistics
            # # we noticed that this form does not contribute much when the superficial statistics learner is NGLCM, but can be helpful in other cases
            # y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(tf.matmul(y_conv_H, y_conv_H, transpose_a=True))),
            #                   y_conv_H, transpose_b=True), y_conv_loss) \
            #               + self.lam * y_conv_H
            # # --------------------------

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))


