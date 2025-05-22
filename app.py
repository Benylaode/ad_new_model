import joblib
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict

# === Semua kolom fitur termasuk fitur numerik ===
# Urutan kolom harus sesuai dengan yang digunakan saat training model
all_columns = [
    'device_type__desktop', 'device_type__mobile', 'device_type__tablet',
    'location__canada', 'location__germany', 'location__india', 'location__uk', 'location__usa',
    'age_group__18-24', 'age_group__25-34', 'age_group__35-44', 'age_group__45-54', 'age_group__55+',
    'gender__female', 'gender__male',
    'content_type__image', 'content_type__text', 'content_type__video',
    'ad_topic__automotive', 'ad_topic__electronics', 'ad_topic__fashion',
    'ad_topic__health', 'ad_topic__travel',
    'ad_target_audience__family oriented', 'ad_target_audience__fitness lovers',
    'ad_target_audience__tech enthusiasts', 'ad_target_audience__travel lovers',
    'ad_target_audience__young adults',
    'engagement_level__commented', 'engagement_level__ignored', 'engagement_level__liked', 'engagement_level__shared',
    'click_through_rate', 'conversion_rate', 'view_time',
    'cost_per_click', 'click_through_rate.1', 'conversion_rate.1', 'calculated_ROI'
]

# === Load Model ===
base_model_0 = joblib.load("base_model_0.pkl")
base_model_1 = joblib.load("base_model_1.pkl")
base_model_2 = joblib.load("base_model_2.pkl")

# === Kelompokkan fitur kategorikal ===
grouped_features = defaultdict(list)
for f in all_columns:
    if '__' in f:
        key, val = f.split('__', 1)
        grouped_features[key.strip()].append(val.strip())

opsi_device = grouped_features["device_type"]
opsi_location = grouped_features["location"]
opsi_age = grouped_features["age_group"]
opsi_gender = grouped_features["gender"]
opsi_content = grouped_features["content_type"]
opsi_topic = grouped_features["ad_topic"]
opsi_engagement = grouped_features["engagement_level"]
opsi_target_audience = grouped_features["ad_target_audience"]
opsi_performance = grouped_features["Performance"]

# === Fungsi validasi input kategorikal ===
def validate_input(options, input_str, feature_name):
    input_lower = input_str.lower()
    valid_options = [opt.lower() for opt in options]
    if input_lower not in valid_options:
        warnings.warn(f"⚠️ '{input_str}' tidak valid untuk {feature_name}. Pilihan: {options}")
        return False
    return True

# === Fungsi input angka ===
def input_float(prompt):
    while True:
        val = input(prompt)
        try:
            return float(val)
        except ValueError:
            print("❌ Harap masukkan angka yang valid!")

def input_int(prompt):
    while True:
        val = input(prompt)
        try:
            return int(val)
        except ValueError:
            print("❌ Harap masukkan angka bulat!")

# === Input fitur kategorikal ===
print("\n🔹 Device Type:")
for i, opt in enumerate(opsi_device, 1): print(f"{i}. {opt}")
device = input("Pilih (masukkan teks): ")
while not validate_input(opsi_device, device, "Device Type"):
    device = input("Coba lagi: ")
f1 = f"device_type__{device.lower()}"

print("\n🔹 Location:")
for i, opt in enumerate(opsi_location, 1): print(f"{i}. {opt}")
location = input("Pilih (masukkan teks): ")
while not validate_input(opsi_location, location, "Location"):
    location = input("Coba lagi: ")
f2 = f"location__{location.lower()}"

print("\n🔹 Age Group:")
for i, opt in enumerate(opsi_age, 1): print(f"{i}. {opt}")
age = input("Pilih (masukkan teks): ")
while not validate_input(opsi_age, age, "Age Group"):
    age = input("Coba lagi: ")
f3 = f"age_group__{age.lower()}"

print("\n🔹 Gender:")
for i, opt in enumerate(opsi_gender, 1): print(f"{i}. {opt}")
gender = input("Pilih (masukkan teks): ")
while not validate_input(opsi_gender, gender, "Gender"):
    gender = input("Coba lagi: ")
f4 = f"gender__{gender.lower()}"

print("\n🔹 Content Type:")
for i, opt in enumerate(opsi_content, 1): print(f"{i}. {opt}")
content = input("Pilih (masukkan teks): ")
while not validate_input(opsi_content, content, "Content Type"):
    content = input("Coba lagi: ")
f5 = f"content_type__{content.lower()}"

print("\n🔹 Ad Topic:")
for i, opt in enumerate(opsi_topic, 1): print(f"{i}. {opt}")
topic = input("Pilih (masukkan teks): ")
while not validate_input(opsi_topic, topic, "Ad Topic"):
    topic = input("Coba lagi: ")
f6 = f"ad_topic__{topic.lower()}"

print("\n🔹 Engagement Level:")
for i, opt in enumerate(opsi_engagement, 1): print(f"{i}. {opt}")
engage = input("Pilih (masukkan teks): ")
while not validate_input(opsi_engagement, engage, "Engagement Level"):
    engage = input("Coba lagi: ")
f7 = f"engagement_level__{engage.lower()}"

print("\n🔹 Target Audience:")
for i, opt in enumerate(opsi_target_audience, 1): print(f"{i}. {opt}")
target_audience = input("Pilih (masukkan teks): ")
while not validate_input(opsi_target_audience, target_audience, "Target Audience"):
    target_audience = input("Coba lagi: ")
f8 = f"ad_target_audience__{target_audience.lower()}"

# === Input fitur numerik ===
print("\n🔢 Masukkan nilai untuk fitur numerik:")
ctr = input_float("🔹 Click Through Rate: ")
conv_rate = input_float("🔹 Conversion Rate: ")
view_time = input_int("🔹 View Time (dalam detik): ")
cpc = input_float("🔹 Cost Per Click: ")
ctr_1 = input_float("🔹 Click Through Rate (versi 2): ")
conv_rate_1 = input_float("🔹 Conversion Rate (versi 2): ")
roi = input_float("🔹 Calculated ROI: ")

selected_features = {f1, f2, f3, f4, f5, f6, f7, f8}
print(selected_features)
feature = [1 if col in selected_features else 0 for col in all_columns if '__' in col]

feature.extend([
    ctr, conv_rate, view_time,
    cpc, ctr_1, conv_rate_1, roi
])

print(len(feature))
print(len(all_columns))

assert len(feature) == len(all_columns), "Jumlah fitur tidak sesuai dengan jumlah kolom"

X_test = pd.DataFrame([feature], columns=all_columns)
y_pred_0 = base_model_0.predict(X_test)
y_pred_1 = base_model_1.predict(X_test)
y_pred_2 = base_model_2.predict(X_test)


y_pred = np.column_stack((y_pred_0, y_pred_1, y_pred_2))
y_pred_ku = np.argmax(np.bincount(y_pred.flatten()))

print("\n=== HASIL PREDIKSI ===")
if y_pred_ku == 0:
    x = "iklan akan memiliki permormence yang low coba perbaiki stategi mu"
elif y_pred_ku == 1:
    x = "iklan akan memiliki performace yang sedang jadi cukup baik namun masih bisa ditingkatkan"
else:
    x = "iklan ini akan sangat baik"
print(f"🎯 Hasil akhir: {x}")