import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from keras_gradient_noise import add_gradient_noise


NoisyAdam = add_gradient_noise(Adam)

x = np.array([
    [0.1],
    [0.5],
    [0.8],
    [0.3],
])
y = np.array([
    [0.9],
    [0.5],
    [0.2],
    [0.7],
])


def test_noisy_optimizer_with_simple_model_training():
    try:
        model = Sequential()
        model.add(Dense(1, input_shape=(1,)))
        model.compile(optimizer=NoisyAdam(), loss='mse')
        model.fit(x, y, epochs=4, batch_size=2, verbose=0)
    except Exception as e:
        pytest.fail("Unexpected MyError: " + str(e))
