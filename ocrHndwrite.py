import cv2  # کتابخانه OpenCV برای پردازش تصویر
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Load and preprocess data (می‌توان این بخش را مشابه قبل نگه داشت)
def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

# 2. Create CNN model (همان بخش ایجاد مدل را نگه دارید)
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 کلاس برای اعداد 0 تا 9
    ])
    return model

# 3. Preprocess and load a new image (تابعی برای پیش‌پردازش تصویر جدید)
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # خواندن تصویر به صورت خاکستری
    img = cv2.resize(img, (28, 28))  # تغییر اندازه به 28x28
    img = img / 255.0  # نرمال‌سازی به محدوده [0, 1]
    img = img.reshape(1, 28, 28, 1)  # تغییر شکل به شکل مورد نیاز مدل
    return img

# 4. Predict a digit from a new image (پیش‌بینی عدد از تصویر جدید)
def predict_digit(model, image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = tf.argmax(prediction[0]).numpy()  # استخراج عدد پیش‌بینی شده
    return predicted_label

# اجرای مراحل
if __name__ == "__main__":
    # 1. Load data and train the model (مرحله آموزش)
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)

    # 2. Predict a new handwritten digit
    image_path = 'path_to_your_image.png'  # مسیر تصویر دست‌نویس خود را اینجا وارد کنید
    predicted_digit = predict_digit(model, image_path)
    print(f'The predicted digit is: {predicted_digit}')
