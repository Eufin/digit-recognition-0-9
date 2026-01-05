import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Muat dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pra proses: normalisasi dan reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Konversi label ke kategorikal
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Buat model CNN yang optimal untuk akurasi maksimal
model = models.Sequential([
    # Blok Conv Pertama - Feature Detection Dasar
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Blok Conv Kedua - Feature Refinement
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Blok Conv Ketiga - Feature Extraction
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Layer Dense untuk Klasifikasi
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

# Compile dengan optimizer yang optimal
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model Summary:")
model.summary()

# Data augmentation yang lebih advanced
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.15,
    fill_mode='nearest'
)

# Callbacks untuk monitoring dan optimization
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_digit_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "="*50)
print("Starting Training...")
print("="*50 + "\n")

# Latih model
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 64,
    callbacks=callbacks,
    verbose=1
)

# Evaluasi model
print("\n" + "="*50)
print("Evaluating Model...")
print("="*50)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✓ Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Final Test Loss: {test_loss:.4f}\n")

# Simpan model terbaik
model.save('digit_model.h5')
print("Model saved as 'digit_model.h5'")

# Simpan model terbaik dari checkpoint juga
import shutil
try:
    shutil.copy('best_digit_model.h5', 'digit_model.h5')
    print("Best model also saved as 'digit_model.h5'")
except:
    pass

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
print("✓ Training history saved as 'training_history.png'")
plt.close()

print("\n" + "="*50)
print("Training Complete!")
print("="*50)