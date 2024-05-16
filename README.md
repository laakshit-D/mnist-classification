# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/eef099d4-ccf0-4148-8d61-3cbe8c06ac37)

## DESIGN STEPS

### STEP 1:
Start by importing all the necessary libraries. And load the Data into Test sets and Training sets.

### STEP 2:
Then we move to normalization and encoding of the data.

### STEP 3:
The Model is then built using a Conv2D layer, MaxPool2D layer, Flatten layer, and 2 Dense layers of 16 and 10 neurons respectively.

### STEP 4:
Finally, we pass handwritten digits to the model for prediction.

## PROGRAM

### Name: LAAKSHIT D
### Register Number: 212222230071
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(single_image,cmap='gray')
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
y_train_onehot.shape
```
```python
single_image = X_train[487]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
```python
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='tanh'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics="accuracy")
```
```python
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
```
```python
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
```python
img = image.load_img('7.png')
type(img)
img = image.load_img('7.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
print(x_single_prediction)
````
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-03-18 143710](https://github.com/laakshit-D/mnist-classification/assets/119559976/0e731f2d-7cf6-4c03-a03b-8420e37a6115)

![Screenshot 2024-03-18 143721](https://github.com/laakshit-D/mnist-classification/assets/119559976/eae63d6c-a4ba-43ba-891c-06fd4b9f71de)

### Classification Report

![Screenshot 2024-03-18 143739](https://github.com/laakshit-D/mnist-classification/assets/119559976/9ab3095f-5583-404f-aaf2-3bd36c3c9b1d)

### Confusion Matrix

![Screenshot 2024-03-18 143730](https://github.com/laakshit-D/mnist-classification/assets/119559976/e3eb21a6-d54d-4d40-ae51-f957edd41c10)

### New Sample Data Prediction
#### INPUT

![7](https://github.com/laakshit-D/mnist-classification/assets/119559976/1ae2679a-c464-44f3-a98f-c20ded85908a)

#### OUTPUT

![Screenshot 2024-03-18 143749](https://github.com/laakshit-D/mnist-classification/assets/119559976/7f35ea55-24c8-45d0-94ca-ead183bb86ae)

![Screenshot 2024-03-18 144043](https://github.com/laakshit-D/mnist-classification/assets/119559976/a0c00fe3-a58d-4599-b7d6-925a42da4433)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
