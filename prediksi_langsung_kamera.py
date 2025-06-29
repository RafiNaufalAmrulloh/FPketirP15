# KODE LENGKAP UNTUK prediksi_langsung_kamera.py (DENGAN 6 KELAS)

import cv2
import tensorflow as tf
import numpy as np
import os

# --- Konfigurasi ---
# == UBAH NAMA MODEL KE VERSI BARU ==
MODEL_PATH = "model_final_berbobot.h5"
IMG_SIZE = 224

# == TAMBAHKAN NAMA KELAS BARU DI AKHIR ==
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR', 'Bukan Retina']

# --- Memuat Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    print(f"Error: Tidak dapat memuat model. Pastikan file '{MODEL_PATH}' ada.")
    exit()

# --- Menyalakan Kamera ---
cap = None
for i in range(3):
    temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"Kamera ditemukan di indeks {i} dengan backend DSHOW.")
        break
    else:
        temp_cap.release()

if not cap:
    print("Error: Kamera tidak dapat ditemukan.")
    exit()

print("\nKamera siap. Arahkan kamera lalu tekan 'SPASI' untuk prediksi atau 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Kamera', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 32:
        print("Memproses gambar...")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        img_normalized = img_array / 255.0
        
        predictions = model.predict(img_normalized)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        # == LOGIKA DIAGNOSIS DIPERBARUI ==
        if predicted_class_name == 'Bukan Retina':
            final_diagnosis = "Gambar Tidak Relevan"
        elif predicted_class_index > 0:
            final_diagnosis = "Terdeteksi Tanda Diabetes"
        else: # predicted_class_index == 0
            final_diagnosis = "Tidak Terdeteksi Tanda Diabetes (Sehat)"

        print(f"Hasil: {predicted_class_name} ({confidence:.2f}%) -> {final_diagnosis}")
        
        text_result = f"{final_diagnosis} ({predicted_class_name}: {confidence:.2f}%)"
        cv2.putText(frame, text_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Kamera', frame)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()