from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D

model = Sequential([
  Conv2D(32, (3,3), input_shape=(28, 28, 3) activation='relu'),
  BatchNormalization(),
  Conv2D(32, (3,3), activation='relu'),
  BatchNormalization(),
  MaxPooling2D(),
  Dense(2, activation='softmax')
])