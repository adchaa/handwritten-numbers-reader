import tensorflow.keras as ker
import numpy as np
# data load and some shit
data = ker.datasets.mnist
(x_train,y_train), (x_ev,y_ev) = data.load_data()

x_train = ker.utils.normalize(x_train, axis=1)
x_ev = ker.utils.normalize(x_ev, axis=1)
x_train = np.expand_dims(x_train,-1)
x_ev = np.expand_dims(x_ev, -1)

#model shit

model = ker.models.Sequential()
model.add(ker.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation=ker.activations.relu ))
model.add(ker.layers.Conv2D(32,(3,3),activation=ker.activations.relu ))
model.add(ker.layers.MaxPool2D(pool_size=(2,2)))
model.add(ker.layers.Conv2D(64,(3,3),activation=ker.activations.relu))
model.add(ker.layers.MaxPool2D(pool_size=(2,2)))
model.add(ker.layers.Flatten())
model.add(ker.layers.Dropout(0.25))
model.add(ker.layers.Dense(10, activation=ker.activations.softmax))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


model.fit(x_train,y_train,epochs=4)
print(model.evaluate(x_ev,y_ev))
model.save("digit.model")
