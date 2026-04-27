import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# PlantVillage dataset path (download from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
possible_paths = ['data/PlantVillage', 'data/val', 'data/PlantVillage/val']
data_dir = next((p for p in possible_paths if os.path.exists(p)), None)
if data_dir is None:
    print("Please download the PlantVillage dataset and place it in data/PlantVillage or data/val.")
    print("Current data/ contents:", os.listdir('data'))
    exit()

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save model
model.save('models/disease_model.h5')
print("Disease model saved to models/disease_model.h5")

# Save class indices
import json
with open('models/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)