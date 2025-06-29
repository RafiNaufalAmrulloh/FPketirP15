# KODE VERSI FINAL DENGAN CLASS WEIGHT UNTUK MENGATASI CLASS IMBALANCE

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os

print("TensorFlow Version:", tf.__version__)

# Persiapan Data (tetap sama)
print("Mempersiapkan data dari semua sumber...")
# ... (Kode data loading dari jawaban sebelumnya tetap sama, saya persingkat di sini agar tidak terlalu panjang) ...
BASE_PATH = 'aptos2019-blindness-detection/'
TRAIN_IMG_PATH = os.path.join(BASE_PATH, 'train_images/')
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
df_retina = pd.read_csv(TRAIN_CSV_PATH)
df_retina['filepath'] = df_retina['id_code'].apply(lambda x: os.path.join(TRAIN_IMG_PATH, x + '.png'))
df_retina = df_retina[['filepath', 'diagnosis']]
NEGATIVE_IMG_PATH = 'data_negatif/'
negative_filepaths = []
if os.path.exists(NEGATIVE_IMG_PATH):
    negative_files = os.listdir(NEGATIVE_IMG_PATH)
    negative_filepaths = [os.path.join(NEGATIVE_IMG_PATH, f) for f in negative_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
df_negative = pd.DataFrame({'filepath': negative_filepaths, 'diagnosis': '5'})
WEBCAM_DATA_PATH = 'data_webcam/'
webcam_data_list = []
if os.path.exists(WEBCAM_DATA_PATH):
    for class_label in os.listdir(WEBCAM_DATA_PATH):
        class_folder_path = os.path.join(WEBCAM_DATA_PATH, class_label)
        if os.path.isdir(class_folder_path):
            for f in os.listdir(class_folder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    webcam_data_list.append({'filepath': os.path.join(class_folder_path, f), 'diagnosis': class_label})
df_webcam = pd.DataFrame(webcam_data_list)
all_dfs = [df_retina, df_negative]
if not df_webcam.empty:
    all_dfs.append(df_webcam)
all_data_df = pd.concat(all_dfs, ignore_index=True)
all_data_df = all_data_df.sample(frac=1, random_state=42).reset_index(drop=True)
all_data_df['diagnosis'] = all_data_df['diagnosis'].astype(str)
train_df, val_df = train_test_split(all_data_df, test_size=0.2, stratify=all_data_df['diagnosis'], random_state=42)

# Augmentasi dan Generator (tetap sama)
IMG_SIZE = 224
BATCH_SIZE = 32
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255., rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest',
    brightness_range=[0.7, 1.3], channel_shift_range=20.0
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, x_col='filepath', y_col='diagnosis',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df, x_col='filepath', y_col='diagnosis',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# === BAGIAN BARU: MENGHITUNG CLASS WEIGHTS ===
print("Menghitung bobot kelas untuk mengatasi data tidak seimbang...")
# Pastikan y_true adalah integer untuk perhitungan
y_true = train_df['diagnosis'].astype(int)
class_weights_array = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_true),
    y=y_true
)
# Ubah menjadi format dictionary yang dibutuhkan TensorFlow
class_weights = dict(enumerate(class_weights_array))
print("Bobot Kelas yang akan digunakan:", class_weights)
# ============================================

# Membangun Model (tetap sama)
base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False
model = tf.keras.models.Sequential([
    base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(6, activation='softmax')
])

# Training TAHAP 1
print("\n--- MEMULAI TAHAP 1: TRAINING AWAL (15 Epoch) ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_awal = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    class_weight=class_weights  # <-- MENERAPKAN CLASS WEIGHT DI SINI
)

# Training TAHAP 2 (Fine-tuning)
print("\n--- MEMULAI TAHAP 2: FINE-TUNING (Hati-hati) ---")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer_fine_tune, loss='categorical_crossentropy', metrics=['accuracy'])
total_epochs = 40
history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=15,
    validation_data=validation_generator,
    class_weight=class_weights  # <-- MENERAPKAN CLASS WEIGHT DI SINI JUGA
)

# Simpan model final
model.save("model_final_berbobot.h5")
print("\nModel final dengan bobot kelas telah disimpan sebagai model_final_berbobot.h5")