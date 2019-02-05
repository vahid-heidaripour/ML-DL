# Tensorflow
Machine learning frameworks like TensorFlow, PaddlePaddle, Torch, Caffe, Keras, and many others can speed up your machine learning development significantly. All of these frameworks also have a lot of documentation, which you should feel free to read. In this assignment, you will learn to do the following in TensorFlow:
    - Initialize variables
    - Start your own session
    - Train algorithms
    - Implement a Neural Network
## Build a neural network in tensorflow
Here we build a neural network using tensorflow. Remember that there are two parts to implement a tensorflow model:
    - Create the computation graph
    - Run the graph
## Output
The output of this program is a neural network with 0.999074 train accuracy, and 0.716667 test accuracy.
![](images/plot.png)
### Insights
    - The model seems big enough to fit the training set well. However, given the difference between train and test accuracy, we could try to add L2 or dropout regularization to reduce overfitting.
    - Think about the session as a block of code to train the model. Each time you run the session on a minibatch, it trains the parameters. In total we have run the session a large number of times (1500 epochs) until we obtained well trained parameters.

## Summary
    - Tensorflow is a programming framework used in deep learning
    - The two main object classes in tensorflow are Tensors and Operators.
    - When you code in tensorflow you have to take the following steps:
        - Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
        - Create a session
        - Initialize the session
        - Run the session to execute the graph
    - You can execute the graph multiple times as you've seen in model()
    - The backpropagation and optimization is automatically done when running the session on the "optimizer" object.
