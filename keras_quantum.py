import tensorflow as tf
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

class example1(object):

	def __init__(self):

		X, y = make_moons(n_samples=200, noise=0.1)
		y_hot = tf.keras.utils.to_categorical(y, num_classes=2)

		layer_1 = tf.keras.layers.Dense(2)
		layer_2 = tf.keras.layers.Dense(2, activation="softmax")

		model = tf.keras.Sequential([layer_1, layer_2])
		model.compile(loss="mae")

		model.fit(X, y_hot, epochs=6, batch_size=5, validation_split=0.25, verbose=2)

		def create_model():

			activation="relu"
			dropout_rate=0.0
			init_mode="uniform"
			weight_constraint=0
			optimizer='adam'
			lr = 0.01
			momentum=0
			model = Sequential()
			model.add(Dense(8,
							input_dim=input_dim, kernal_initializer=init_mode,
							activation=activation,
							kernal_constraint=maxnorm(weight_constraint)))

		np.random.seed(42)
		tf.random.set_seed(42)

		plt.axis("off")
		plt.scatter(X[:, 0], X[:, 1])
		plt.show()


		self.model = model

	def returnFunc(self):
		return self.model

x = example1()
model = x.returnFunc()
print(model.to_json())
# print(dir(model))

import pennylane as qml
from pennylane import QubitDevice

class MyDevice(QubitDevice):
	"""MyDevice docstring"""
	name = "My custom device"
	short_name = "example.mydevice"
	pennylane_requires = "0.1.0"
	version = "0.0.1"
	author = "Dennis Biber"
	operations = ("CNOT", "PauliX")
	observables = ("PauliZ", "PauliX", "PauliY")

	def __init__(self, shots=1024, hardware_option=None):
		super().__init__(wires=24, shots=shots, analytic=False)
		self.hardware_option = hardware_option or hardware_defaults

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