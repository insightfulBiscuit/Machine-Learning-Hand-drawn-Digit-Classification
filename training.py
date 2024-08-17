import tensorflow as tf             #ML

#loading dataset
mnist = tf.keras.datasets.mnist

#split dataset into training and testing data
(training_image, training_label), (testing_image, testing_label) = mnist.load_data()

#normalizing (from 0-255, to 0-1)
training_image = tf.keras.utils.normalize(training_image, axis=1)
testing_image = tf.keras.utils.normalize(testing_image, axis=1)

#creating a basic model
model = tf.keras.models.Sequential()

#adding input layer (28x28 because of mnist handwritten diggit size)
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#adding dense layer (128 neurons)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
#adding output layer (softmax scales all outputs to add up to 1)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#compiling model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training model
model.fit(training_image, training_label, epochs=3)

model.save('handwritten_digits_1.0')