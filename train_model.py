import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json

# 🔥 Data generators (strong augmentation)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.7, 1.3]
)

val_gen = ImageDataGenerator(rescale=1./255)

# 📂 Load dataset
train_data = train_gen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    'dataset/val',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# 💾 Save class labels
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# 🧠 Load MobileNetV2
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# ❄️ Freeze most layers
base_model.trainable = False

# 🔓 Unfreeze last few layers (important for your dataset)
for layer in base_model.layers[-20:]:
    layer.trainable = True

# 🏗️ Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),   # helps reduce overfitting
    layers.Dense(train_data.num_classes, activation='softmax')
])

# ⚙️ Compile (low learning rate = stable training)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🛑 Early stopping (stops if no improvement)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 📉 Reduce LR if stuck
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2
)

# 🚀 Train
model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop, reduce_lr]
)

# 💾 Save model
model.save("hair_disease_model.h5")

print("✅ Model trained successfully with improved accuracy!")