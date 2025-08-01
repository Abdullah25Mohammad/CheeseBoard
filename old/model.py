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

def to_categorical_binary(y):
    y_cat = np.zeros((len(y), 2))  # [white, black]
    y_cat[y == 1] = [1, 0]
    y_cat[y == -1] = [0, 1]
    return y_cat

y = to_categorical_binary(y)

current_player,turns_played = X['current_player'], X['turns_played']
# Drop the columns used for current player and turns played
X = X.drop(columns=['current_player', 'turns_played'])

# Reshape features for CNN input
X = X.values.reshape(-1, 8, 8, 1)

# Channelize the features
X_channelized = np.zeros((X.shape[0], 8, 8, 12), dtype=np.uint8)

# Mapping piece values to channel indices
channel_map = {
    1: 0,    # White Pawn
    2: 1,    # White Knight
    3: 2,    # White Bishop
    4: 3,    # White Rook
    5: 4,    # White Queen
    6: 5,    # White King
    -1: 6,   # Black Pawn
    -2: 7,   # Black Knight
    -3: 8,   # Black Bishop
    -4: 9,   # Black Rook
    -5: 10,  # Black Queen
    -6: 11   # Black King
}

# Flatten the last dimension (from shape [N,8,8,1] to [N,8,8])
X_flat = X.squeeze(-1)

# Vectorized channelization
for piece_val, channel_idx in channel_map.items():
    mask = (X_flat == piece_val)
    X_channelized[mask, channel_idx] = 1

# Final input to model
X = X_channelized

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()

# Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(8, 8, 12)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))  # 8x8 → 4x4

# Layer 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))  # 4x4 → 2x2

# Layer 3
model.add(Conv2D(128, (2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Flatten and Dense
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))  # Output: [white_prob, black_prob]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('chess_cnn_model.h5')






