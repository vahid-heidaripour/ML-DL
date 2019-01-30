# Optimization Methods
Optimization methods can speed up learning and perhaps even get you to a better final value for the cost function. Having a good optimization algorithm can be the difference between waiting days vs. just a few hours to get a good result.
## Model with different optimization algorithms
Here we have a model with different optimization algorithms. We implemented a 3-layer neural network. Now we want ot train it with:
	- Mini-batch Gradient Descent: it calls function: update_parameters_with_gd()
	- Mini-batch Momentum: it calls functions: initialize_velocity() and update_parameters_with_momentum()
	- Mini-batch Adam: it calls functions: initialize_adam() and update_parameters_with_adam()

### 1 - Mini-batch Gradient descent
Uncomment the code below to run it and see how the model does the mini-batch gradient descent.
```
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

### 2 - Mini-batch gradient descent with momentum
Uncomment the code below to run it and see how the modle does with momentum.Because this example is relatively simple, the gains from using momemtum are small; but for more complex problems you might see bigger gains.
```
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

### 3 - Mini-batch with Adam mode
Uncomment the code below to run it and see how the modle does with Adam.
```
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

### Summary
| **optimization method** | **accuracy** | **cost shape** |
| Gradient descent | 79.7% | oscillations |
| Momentum | 79.7% | oscillations | 
| Adam | 94% | smoother |

Momentum usually helps, but given the small learning rate and the simplistic dataset, its impact is almost negligeable. Also, the huge oscillations you see in the cost come from the fact that some minibatches are more difficult thans others for the optimization algorithm.
Adam on the other hand, clearly outperforms mini-batch gradient descent and Momentum. If you run the model for more epochs on this simple dataset, all three methods will lead to very good results. However, you've seen that Adam converges a lot faster.

