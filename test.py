import tensorflow.keras as tf
import tensorflow


data = tf.datasets.mnist
(x_train,y_train), (x_ev,y_ev) = data.load_data()
model = tf.models.Sequential()
x_train = tf.utils.normalize(x_train, axis=1)
x_ev = tf.utils.normalize(x_ev, axis=1)


model.add(tf.layers.Flatten(input_shape=(28,28)))
model.add(tf.layers.Dense(128,activation=tensorflow.nn.relu))
model.add(tf.layers.Dense(128,activation=tensorflow.nn.relu))
model.add(tf.layers.Dense(128,activation=tensorflow.nn.relu))
model.add(tf.layers.Dense(10,activation=tensorflow.nn.softmax))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=3)
print(model.evaluate(x_ev,y_ev))
model.save("digit.model")



