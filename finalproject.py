import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tkinter import Tk, filedialog

# 1. تابع برای استخراج ویژگی‌های صوتی (MFCC)
def extract_features(audio_files, sr=16000, n_mfcc=13):
    features = []
    for audio_path in audio_files:
        audio, sr = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = np.mean(mfccs.T, axis=0)
        features.append(mfccs)
    return np.array(features)

# 2. تابع برای ایجاد مدل LSTM
def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. تابع برای تبدیل پیش‌بینی‌ها به متن
def predictions_to_text(predictions, label_encoder):
    return label_encoder.inverse_transform([np.argmax(pred) for pred in predictions])

# 4. تابع اصلی
def main():
    # 4.1. دریافت مسیر دیتاست
    root = Tk()
    root.withdraw()
    dataset_path = filedialog.askdirectory(title="لطفا مسیر دیتاست را انتخاب کنید")
    if not dataset_path:
        print("هیچ مسیری انتخاب نشد! برنامه پایان یافت.")
        return
    
    print(f"مسیر دیتاست انتخاب شده: {dataset_path}")
    
    # 4.2. بارگذاری فایل‌های صوتی و برچسب‌ها
    audio_files = []
    labels = []
    for root_dir, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root_dir, file))
                labels.append(os.path.basename(root_dir))  # نام پوشه به عنوان برچسب
    
    if not audio_files:
        print("هیچ فایل صوتی با پسوند .wav در مسیر انتخابی یافت نشد!")
        return
    
    print(f"تعداد فایل‌های صوتی یافت شده: {len(audio_files)}")
    
    # 4.3. استخراج ویژگی‌ها
    features = extract_features(audio_files)
    
    # 4.4. تبدیل برچسب‌ها به عدد
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # 4.5. تقسیم داده‌ها به Train، Validation و Test
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # تغییر شکل برای LSTM
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # 4.6. ایجاد و آموزش مدل
    input_shape = (X_train.shape[1], 1)
    num_classes = len(label_encoder.classes_)
    model = build_model(input_shape, num_classes)
    
    # استفاده از EarlyStopping برای جلوگیری از Overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("\nشروع آموزش مدل...")
    history = model.fit(
        X_train, y_train,
        epochs=50,  # تعداد دوره‌ها
        batch_size=32,
        validation_split=0.2,  # استفاده از Validation Set
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 4.7. ارزیابی مدل روی Test Set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nدقت مدل روی Test Set: {test_accuracy * 100:.2f}%")
    
    # 4.8. تبدیل صوت به متن
    predictions = model.predict(X_test)
    predicted_texts = predictions_to_text(predictions, label_encoder)
    
    # 4.9. ذخیره نتایج در یک فایل
    output_file = os.path.join(dataset_path, "predictions.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for true_label, pred_text in zip(y_test, predicted_texts):
            true_text = label_encoder.inverse_transform([true_label])[0]
            f.write(f"برچسب واقعی: {true_text}, برچسب پیش‌بینی‌شده: {pred_text}\n")
    
    print(f"نتایج در فایل {output_file} ذخیره شد.")

# 5. اجرای تابع اصلی
if __name__ == "__main__":
    main()