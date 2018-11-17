# Learned the basic classification given in Tensorflow tutorial...

# Reference: https://www.tensorflow.org/tutorials/keras/basic_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


#Dataset used is Fashion MNIST Dataset.
# this is a good starting point to train, test and debug code

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Print the train_image and train_label to see how they're represented..
#print(train_images[0])
#print(train_labels[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Print the shape of training_image.
#print(train_images.shape)

#print("Size of training data=",len(train_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

#cast from int to float and divide by 255
train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


#Create the model specifying two layers and 128 neurons in one layer and 10 neurons in second layer..
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#How the model would progress specifying the accuracy, loss and optimizer()/updates
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model based on train_images and train_labels.
model.fit(train_images, train_labels, epochs=5) #Number of passes, epoch = 5

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# test this learnt model on a test image.
image = test_images[0]

# print the shape of the image
print(image.shape)


#Add this image to batch (We are testing on a single image
image = (np.expand_dims(image,0))

#print the shape of the image now...
print(image.shape)

single_test_prediction = model.predict(image)

print(np.argmax(single_test_prediction[0]))

print("The test output label=",test_labels[0])


# make the predictions of model on the test images.....
predictions = model.predict(test_images)

# Now prediction is a numpy array consisting of confidence of each class.

np.argmax(predictions[0]) #Find or predict the class with maximum confidence.

print("test_label=",test_labels[0])
