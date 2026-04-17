import tensorflow as tf
from tensorflow.keras import models, layers

# 1. Configuration
IMAGE_SIZE = 128  # Using 128 for faster training and lower memory usage
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 15       # 15-20 epochs are sufficient for 6 classes

# 2. Load Dataset (Loading 6 classes from your 'dataset' folder)
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Automatically count the number of classes (this will detect all 6 folders)
class_names = dataset.class_names
num_classes = len(class_names) 
print(f"Detected classes: {class_names}")

# 3. CNN Model (Modified for 6 classes)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Automatically sets output to 6 classes
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# 4. Start Training
model.fit(dataset, epochs=EPOCHS)

# 5. Save Model
model.save("leaf_model.h5") 

print("Congratulations! The new model has been created and saved successfully.")