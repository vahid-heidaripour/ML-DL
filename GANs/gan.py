import tensorflow as tf
import numpy as np
from create_training_data import *
import seaborn as sb
import matplotlib.pyplot as plt

def getSampleZ(m, n):
    return np.random.uniform(-1, 1, size=[m, n])

def generator(Z, hsize = [16, 16], reuse = False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.relu)
        out = tf.layers.dense(h2, 2)

    return out

def discriminator(X, hsize = [16, 16], reuse = False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 2)
        out = tf.layers.dense(h3, 1)

    return out, h3

X = tf.placeholder(tf.float32, [None, 2])
Z = tf.placeholder(tf.float32, [None, 2])

gSample = generator(Z)
rlogits, rrep = discriminator(X)
flogits, grep = discriminator(gSample, reuse=True)

discLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rlogits,
                                                                  labels=tf.ones_like(rlogits)) +
                          tf.nn.sigmoid_cross_entropy_with_logits(logits=flogits,
                                                                  labels=tf.zeros_like(flogits)))
genLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flogits,
                                                                 labels=tf.ones_like(flogits)))

genVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
discVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

genStep = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(genLoss, var_list=genVars)
discStep = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(discLoss, var_list=discVars)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batchSize = 256
ndSteps = 10
ngSteps = 10

xPlot = generateSample(n=batchSize)

f = open('lossLogs.csv', 'w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

for i in range(5001):
    xBatch = generateSample(n=batchSize)
    zBatch = generateSample(batchSize, 2)

    for _ in range(ndSteps):
        _, dloss = sess.run([discStep, discLoss], feed_dict={X: xBatch, Z: zBatch})

    rrepDstep, grepDstep = sess.run([rrep, grep], feed_dict={X: xBatch, Z: zBatch})

    for _ in range(ngSteps):
        _, gloss = sess.run([genStep, genLoss], feed_dict={Z: zBatch})

    rrepGstep, grepGstep = sess.run([rrep, grep], feed_dict={X: xBatch, Z: zBatch})

    print "Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss)

    if i % 10 == 0:
        f.write("%d,%f,%f\n" % (i, dloss, gloss))

    if i % 1000 == 0:
        plt.figure()
        gPlot = sess.run(gSample, feed_dict={Z: zBatch})
        xax = plt.scatter(xPlot[:, 0], xPlot[:, 1])
        gax = plt.scatter(gPlot[:, 0], gPlot[:, 1])

        plt.legend((xax, gax), ("Real Data","Generated Data"))
        plt.title('Samples at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig('./plots/iterations/iteration_%d.png' % i)
        plt.close()

        plt.figure()
        rrd = plt.scatter(rrepDstep[:, 0], rrepDstep[:, 1], alpha=0.5)
        rrg = plt.scatter(rrepGstep[:, 0], rrepGstep[:, 1], alpha=0.5)
        grd = plt.scatter(grepDstep[:, 0], grepDstep[:, 1], alpha=0.5)
        grg = plt.scatter(grepGstep[:, 0], grepGstep[:, 1], alpha=0.5)

        plt.legend((rrd, rrg, grd, grg), ("Real Data Before G step","Real Data After G step",
                               "Generated Data Before G step","Generated Data After G step"))

        plt.title('Transformed Features at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig('./plots/features/feature_transform_%d.png' % i)
        plt.close()

        plt.figure()
        rrdc = plt.scatter(np.mean(rrepDstep[:, 0]), np.mean(rrepDstep[:, 1]), s=100, alpha=0.5)
        rrgc = plt.scatter(np.mean(rrepGstep[:, 0]), np.mean(rrepGstep[:, 1]), s=100, alpha=0.5)
        grdc = plt.scatter(np.mean(grepDstep[:, 0]), np.mean(grepDstep[:, 1]), s=100, alpha=0.5)
        grgc = plt.scatter(np.mean(grepGstep[:, 0]), np.mean(grepGstep[:, 1]), s=100, alpha=0.5)

        plt.legend((rrdc, rrgc, grdc, grgc), ("Real Data Before G step", "Real Data After G step",
                                              "Generated Data Before G step", "Generated Data After G step"))

        plt.title('Centroid of Transformed Features at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig('./plots/features/feature_transform_centroid_%d.png' % i)
        plt.close()

f.close()
