"""
Recurring Neural Network

input > weight > Hidden Layer1  (Activation function) > weight >
Hidden Layer2  (Activation function) >output

compare output to intended output > cost or loss function
optimization function > minimize cost

back-propagation

Feed Forward + back-propagation = epoch (1 cycle)

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

hm_epochs = 5
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'Biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2]) # prem
    x = tf.reshape(x, [-1, chunk_size])  # single line till chunk size of x ->next
    x = tf.split(x, n_chunks, axis=0)  # value, no of splits, axis

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)

    # Creates a recurrent neural network specified by RNNCell
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['Biases']

    return output


def train_neural_network(x):

    prediction = recurrent_neural_network_model(x)

    # comparing output with actual output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    # AdamOptimizer(learning rate =0.0001)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycle of (feedforwrd + backpropogation)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples/batch_size)):

                epoch_x, epoch_y = mnist.train.next_batch(batch_size)  # data, labels
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch ', epoch, ' Completed ', hm_epochs, ' loss ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


train_neural_network(x)

# 55
