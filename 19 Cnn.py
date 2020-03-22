#  convolution neural Network

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#  one hot = one device is on and other is off
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

n_class = 10
batch_size = 128  # batch of 100 feature at one time in a Network

# height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W): # Computes a 2-D convolution
    # input,filters ,determines how much the window shifts by in each of the dimensions(one in this)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpoll2d(x):  # Performs the max pooling on the input.
    #                        size of window 2*2, movement in window2*2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def neural_network_model(x):
    # matrix of random of (784 * 500)
    # tf.random_normal() : Output random value from normal distributions

    #           5 x 5 convolution, 1 input image, 32 outputs
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),

               # 5x5 conv, 32 inputs, 64 outputs
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),

               # fully connected, 7*7*64 inputs, 1024 outputs
               'W_Fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),  # fully connected

               # 1024 inputs, 10 outputs (class prediction)
               'out': tf.Variable(tf.random_normal([1024, n_class]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_Fc': tf.Variable(tf.random_normal([1024])),  # fully connected
              'out': tf.Variable(tf.random_normal([n_class]))}

    # Reshape input to a 4D tensor
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+biases['b_conv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpoll2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'])+biases['b_conv2'])
    conv2 = maxpoll2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])

    fc = tf.nn.relu(tf.matmul(fc, weights['W_Fc']))+biases['b_Fc']
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']+biases['out'])
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)

    # comparing output with actual output  , function operates on the unscaled output of earlier layers
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    # AdamOptimizer(learning rate =0.0001)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycle of (feedforwrd + backpropogation)
    howmany_epoch = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(howmany_epoch):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)  # data, labels
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})  # c : cost
                epoch_loss += c
            print('Epoch ', epoch, ' Completed ', howmany_epoch, ' loss ', epoch_loss)

        #  tf.argmax : return index of max value
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # tf.reduce_mean : compute the mean , tf.cast : change type (to float)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # eval: evaluate
        print('Accuracy ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)


# 58