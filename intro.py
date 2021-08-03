import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

train = pd.read_csv('data/train.csv')
sys.getsizeof(train)/1e+6 # Get size in MB
# 263.8 MB
train.columns # Get column names
# Get number of rows and columns
train.shape
# (42000, 785)
# First column is ID, the rest are pixels (28x28 = 784 pixels in total)

# Split the training data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train.iloc[:,1:], train.iloc[:,0], test_size=0.2, random_state=42)
X_train.shape
# (33600, 784)
all(X_train.columns == X_valid.columns)
# True
# Get type of created objects
type(X_train)
type(X_valid)
type(y_train)
type(y_valid)

# Some interesting models are KNN, SVM and CNN and finally we can use autoML

# Using K-nearest neighbors ====
# Create K-nearest neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2) # p=2 for Euclidean distance
# Train the classifier
knn.fit(X_train, y_train)
# Predict the labels of the validation set
y_pred = knn.predict(X_valid)
# Get the accuracy of the classifier
knn.score(X_valid, y_valid)
# 0.9666666666666667

# Predict the labels for the prediction set
pred_set = pd.read_csv('data/test.csv')
pred_set.shape
final_pred = knn.predict(pred_set)

# Save KNN predictions to csv file
results_knn = pd.DataFrame({'ImageId': range(1, len(final_pred)+1), 'Label': final_pred})
# Create subdirectory results
os.mkdir('results')
results_knn.to_csv('results/submission-knn1.csv', index=False)
# Score Kaggle = 0.96525

# Using Convolutional Neural Networks ====

# Create CNN classifier
# Use tensoflow as backend
X_train_norm = tf.keras.utils.normalize(X_train, axis=1)
X_valid_norm = tf.keras.utils.normalize(X_valid, axis=1)
# Tested also without normalization, results were subotimal
# Train the classifier
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
cnn = tf.keras.Sequential()
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
cnn.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
cnn.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(np.asarray(X_train_norm), np.asarray(y_train), epochs=5)
# Predict the labels of the validation set
y_pred_cnn = cnn.predict(np.asarray(X_valid_norm))
# Get the accuracy of the classifier
cnn.evaluate(np.asarray(X_valid_norm), np.asarray(y_valid))
# loss: 0.1078 - accuracy: 0.9694
y_pred_cnn_f = cnn.predict_classes(np.asarray(X_valid_norm))

# Predict the labels for the prediction set
pred_set = pd.read_csv('data/test.csv')
pred_set_norm = tf.keras.utils.normalize(pred_set, axis=1)
final_pred_cnn = cnn.predict(np.asarray(pred_set_norm))
final_pred_cnn_f = cnn.predict_classes(np.asarray(pred_set_norm))

# Save CNN predictions to csv file
results_cnn = pd.DataFrame({'ImageId': range(1, len(final_pred_cnn_f)+1), 'Label': final_pred_cnn_f})
# Create subdirectory results
if not os.path.exists('results'):
    os.mkdir('results')
results_cnn.to_csv('results/submission-cnn1.csv', index=False)
# Score Kaggle = 0.96446