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

# for col in X.columns:
#     X[col] = normalize(X[col])

current_player,turns_played = X['current_player'], X['turns_played']
# Drop the columns used for current player and turns played
X = X.drop(columns=['current_player', 'turns_played'])

# Reshape features for CNN input
X = X.values.reshape(-1, 8, 8, 1)  # Reshape to (samples, height, width, channels)

# One hot encode to a 12 channel representation
X_flat = X.squeeze(-1)
# Mapping: values -6 to -1 → indices 0 to 5, values 1 to 6 → indices 6 to 11
value_to_index = np.zeros(13, dtype=int)  # values from -6 to 6 mapped to [0...12], with 0 unused
value_to_index[0:6] = np.arange(6)         # -6 to -1 → 0 to 5
value_to_index[7:] = np.arange(6, 12)      # 1 to 6 → 6 to 11
# Shift values from [-6,6] → [0,12] for indexing
X_shifted = X_flat + 6  # Now values are in [0,12
# Create empty encoded array
X_encoded = np.zeros((*X_flat.shape, 12), dtype=np.uint8)  # (100000, 8, 8, 12)
# Get mask of nonzero values
nonzero_mask = X_flat != 0
# Get indices where values are nonzero
samples, rows, cols = np.where(nonzero_mask)
vals = X_flat[nonzero_mask]
channel_indices = value_to_index[vals + 6]
# Set the appropriate channel to 1
X_encoded[samples, rows, cols, channel_indices] = 1

# Reshape back to (samples, height, width, channels)
X = X_encoded.reshape(-1, 8, 8, 12)




# Convert labels to numpy array
y = y.values


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()

# Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 12)))
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
model.add(Dense(1, activation='linear'))  # Regression output for position evaluation


import tf_keras.backend as K

def range_based_accuracy(y_true, y_pred):
    black_win = K.cast(y_pred < 0.0, dtype='float32')
    white_win = 1.0 - black_win
    y_pred_class = -1.0 * black_win + 1.0 * white_win
    return K.mean(K.equal(y_true, y_pred_class))




# Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[range_based_accuracy])

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('chess_cnn_model.h5')






