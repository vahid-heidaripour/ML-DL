import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mist = input_data.read_data_sets("MNIST_data")

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