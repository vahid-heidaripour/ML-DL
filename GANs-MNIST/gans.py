import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data")

def generator(Z, hsize = [128, 128], reuse = False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.relu)
        out = tf.layers.dense(h2, 784, activation=tf.nn.tanh)

        return out

def discriminator(X, hsize = [128, 128], reuse = False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.relu)
        logits = tf.layers.dense(h2, 1)
        out = tf.sigmoid(logits)

        return out, logits

tf.reset_default_graph()

realImages = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(Z)

dOutReal, dLogitsReal = discriminator(realImages)
dOutFake, dLogitsFake = discriminator(G, reuse=True)

def lossFunc(logitsIn, labelsIn):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitsIn, labels=labelsIn))

dRealLoss = lossFunc(dLogitsReal, tf.ones_like(dLogitsReal) * 0.9)
dFakeLoss = lossFunc(dLogitsFake, tf.zeros_like(dLogitsReal))
dLoss = dRealLoss + dFakeLoss

gLoss = lossFunc(dLogitsFake, tf.ones_like(dLogitsFake))

learningRate = 0.001

tvars = tf.trainable_variables()
dvars = [var for var in tvars if 'GAN/Discriminator' in var.name]
gvars = [var for var in tvars if 'GAN/Generator' in var.name]

dTrainer=tf.train.AdamOptimizer(learningRate).minimize(dLoss, var_list=dvars)
gTrainer=tf.train.AdamOptimizer(learningRate).minimize(gLoss, var_list=gvars)

batchSize = 100
epochs = 100
init = tf.global_variables_initializer()

samples = []

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        numBatches = mnist.train.num_examples//batchSize

        for i in range(numBatches):
            batch = mnist.train.next_batch(batchSize)
            batchImages = batch[0].reshape((batchSize, 784))
            batchImages = batchImages * 2 - 1
            batchZ = np.random.uniform(-1, 1, size=(batchSize, 100))

            _ = sess.run(dTrainer, feed_dict={realImages: batchImages, Z: batchZ})
            _ = sess.run(gTrainer, feed_dict={Z: batchZ})

        print("epoch{}".format(epoch))

        sampleZ = np.random.uniform(-1, 1, size=(1, 100))
        genSample = sess.run(generator(Z, reuse=True), feed_dict={Z: sampleZ})

        samples.append(genSample)

plt.imshow(samples[0].reshape(28, 28))
plt.imshow(samples[99].reshape(28, 28))