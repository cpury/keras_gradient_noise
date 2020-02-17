# keras_gradient_noise

Simple way to add gradient noise to any Keras / TensorFlow-Keras optimizer.

Install via: `pip install keras_gradient_noise`


## Gradient Noise

Introduced by
["Adding Gradient Noise Improves Learning for Very Deep Networks" (Neelakantan et al 2015)](https://arxiv.org/abs/1511.06807),
the idea is to add a bit of decaying Gaussian noise to your gradients before
each update step. This is shown to reduce overfitting and training loss.

Equation 1 of the paper defines two parameters for the method:

* η defines the total amount of noise (recommended to be one of {0.01, 0.3, 1.0})
* γ defines the decay rate of the noise (recommended to be 0.55)


## How to use in your code

Simply wrap your optimizer class with the provided `add_gradient_noise()`
function:

```python
from keras.optimizers import Adam
from keras_gradient_noise import add_gradient_noise

# ...

NoisyAdam = add_gradient_noise(Adam)

model.compile(optimizer=NoisyAdam())
```

Note the use of brackets. `add_gradient_noise()` expects a Keras-compatible
optimizer *class*, not an *instance* of one.

You can adjust the two parameters η and γ via initialization arguments. They
have the following default values:

```python
NoisyOptimizer(noise_eta=0.3, noise_gamma=0.55)
```


## Keras vs TF.Keras

The package tries to be smart about whether to use `tf.keras` or standalone `keras`.
If you get an error in your case, try passing a specific Keras-module to the
`add_gradient_noise` function. E.g.

```
import keras

...

add_gradient_noise(MyOptim, keras=keras)
```


## Feedback, contributions, etc.

Please don't hesitate to reach out via GitHub issues or a quick email! Thanks!
