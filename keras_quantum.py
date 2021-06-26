import tensorflow as tf

# tf.keras/backend.set_floatx("float64")

# layer_1 = tf.keras.layers.Dense(2)
# layer_2 = tf.keras.layers.Dense(2, activation="softmax")

# model = tf.keras.Sequential([layer_1, layer_2])
# model.compile(loss="mao")

# model.fit()

# def create_model():

# 	activation="relu"
# 	dropout_rate=0.0
# 	init_mode="uniform"
# 	weight_constraint=0
# 	optimizer='adam'
# 	lr = 0.01
# 	momentum=0
# 	model = Sequential()
# 	model.add(Dense(8
# 					input_dim=input_dim, kernal_initializer=init_mode,
# 					activation=activation,
# 					kernal_constraint=maxnorm(weight_constraint)))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

#// Set Random seeds
np.random.seed(42)
tf.random.set_seed(42)

X, y = make_moons(n_samples=200, noise=0.1)
y_hot = tf.keras.utils.to_categorical(y, num_classes=2)

c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()

import pennylane as qml

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
	qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
	qml.templates.BasicEntaglerLayers(weights, wires=range(n_qubits))
	return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

clayer_1 = tf.keras.layers.Dense(2)
clayer_2 = tf.keras.layers.Dense(2, activation="softmax")
model = tf.keras.models.Sequential([clayer_1, qlayer, clayer_2])

#// Adjust learning rate to tweek accuracy
opt = tf.keras.optimizer.SGD(learning_rate=0.2)
model.compile(opt, loss="mae", metrics=["accuracy"])

fitting = model.fit(X, y_hot, epochs=6, batch_size=5, validation_split=0.25, verbose=2)