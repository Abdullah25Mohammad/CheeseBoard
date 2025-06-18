from tf_keras.models import Sequential
from tf_keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.regularizers import l2
from tf_keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split


# Load training data
df = pd.read_csv('data/training_data.csv')

# Split data into features and labels
X = df.drop(columns=['winner'])
y = df['winner']

# Normalize features
def normalize(col):
    """Normalize a column by dividing by the maximum value in that column."""
    max_value = col.max()
    if max_value > 0:
        return col / max_value
    return col

for col in X.columns:
    X[col] = normalize(X[col])

current_player,turns_played = X['current_player'], X['turns_played']
# Drop the columns used for current player and turns played
X = X.drop(columns=['current_player', 'turns_played'])

# Reshape features for CNN input
X = X.values.reshape(-1, 8, 8, 1)  # Reshape to (samples, height, width, channels)
# Convert labels to numpy array
y = y.values


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()

# Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))  # 8x8 → 4x4

# Layer 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))  # 4x4 → 2x2

# Layer 3
model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))  # 2x2 → 2x2
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Flatten and Dense
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('chess_cnn_model.h5')






