# Deep Neural Network Application Image Classification
This is a code to build and apply a deep neural network to supervised learning.

## 1 - Architecture of the model
Here we built two different models:
	- A 2-layer neural network
	- An L-layer deep neural network

## General methodology
As usual we will follow the Deep Learning methodology to build the model:
1. Initialize parameters / Define hyperparameters
2. Loop for num_iterations:
    a. Forward propagation
    b. Compute cost function
    c. Backward propagation
    d. Update parameters (using parameters, and grads from backprop) 
4. Use trained parameters to predict labels

## Run Two-layer neural network
Uncomment the code below right after two_layer_model function in dnn.py:
```
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
```
The output is:

Cost after iteration 0: 0.693049735659989

Cost after iteration 100: 0.6464320953428849

Cost after iteration 200: 0.6325140647912678

Cost after iteration 300: 0.6015024920354665

Cost after iteration 400: 0.5601966311605747

Cost after iteration 500: 0.515830477276473

Cost after iteration 600: 0.47549013139433266

Cost after iteration 700: 0.43391631512257495

Cost after iteration 800: 0.4007977536203886

Cost after iteration 900: 0.3580705011323798

Cost after iteration 1000: 0.3394281538366413

Cost after iteration 1100: 0.3052753636196265

Cost after iteration 1200: 0.2749137728213018

Cost after iteration 1300: 0.24681768210614863

Cost after iteration 1400: 0.1985073503746611

Cost after iteration 1500: 0.17448318112556635

Cost after iteration 1600: 0.17080762978096914

Cost after iteration 1700: 0.11306524562164712

Cost after iteration 1800: 0.0962942684593716

Cost after iteration 1900: 0.08342617959726877

Cost after iteration 2000: 0.07439078704319092

Cost after iteration 2100: 0.06630748132267941

Cost after iteration 2200: 0.05919329501038178

Cost after iteration 2300: 0.05336140348560562

Cost after iteration 2400: 0.04855478562877025

![](images/figure_1.png)

```
predictions_train = predict(train_x, train_y, parameters)
```
Accuracy: 0.9999999999999998

```
predictions_test = predict(test_x, test_y, parameters)
```
Accuracy: 0.72

## Run L-layer neural network
Uncomment the code below right after the L_layer_model function in dnn.py
```
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
```
The output is:

Cost after iteration 0: 0.771749

Cost after iteration 100: 0.672053

Cost after iteration 200: 0.648263

Cost after iteration 300: 0.611507

Cost after iteration 400: 0.567047

Cost after iteration 500: 0.540138

Cost after iteration 600: 0.527930

Cost after iteration 700: 0.465477

Cost after iteration 800: 0.369126

Cost after iteration 900: 0.391747

Cost after iteration 1000: 0.315187

Cost after iteration 1100: 0.272700

Cost after iteration 1200: 0.237419

Cost after iteration 1300: 0.199601

Cost after iteration 1400: 0.189263

Cost after iteration 1500: 0.161189

Cost after iteration 1600: 0.148214

Cost after iteration 1700: 0.137775

Cost after iteration 1800: 0.129740

Cost after iteration 1900: 0.121225

Cost after iteration 2000: 0.113821

Cost after iteration 2100: 0.107839

Cost after iteration 2200: 0.102855

Cost after iteration 2300: 0.100897

Cost after iteration 2400: 0.092878

![](images/figure_2.png)

```
pred_train = predict(train_x, train_y, parameters)
```
Accuracy: 0.9856459330143539

```
pred_test = predict(test_x, test_y, parameters)
```
Accuracy: 0.8

Congrats! It seems that the 5-layer neural network has better performance (80%) than the 2-layer neural network (72%) on the same test set.

## Result Analysis
A few type of images the model tends to do poorly on include:
	- Cat body in an unusual position
	- Cat appears against a background of a similar color
	- Unusual cat color and species
	- Camera Angle
	- Brightness of the picture
	- Scale variation (cat is very large or small in image)
