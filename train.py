import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

base_dir = 'dataset'

# Data Augmentation ছাড়া সাধারণ রিস্কেল
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# মডেল আর্কিটেকচার
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'), # অতিরিক্ত লেয়ার ভালো রেজাল্টের জন্য
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training is starting...")
# ক্লাস ইনডেক্স চেক করা (Diseased=0, Healthy=1 কি না তা নিশ্চিত হওয়া)
print("Class Indices:", train_gen.class_indices)

model.fit(train_gen, epochs=30, validation_data=val_gen)

model.save('leaf_model.h5')
print("Success! leaf_model.h5 has been created.")