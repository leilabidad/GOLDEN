import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Load and preprocess data
def load_and_preprocess_data():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Reshape data to be compatible with CNN
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Normalize data to the range [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

# 2. Create CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
    ])
    return model

# 3. Compile and train the model
def compile_and_train_model(model, train_images, train_labels):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model with training data
    model.fit(train_images, train_labels, epochs=5, 
              validation_data=(test_images, test_labels))

# 4. Evaluate the model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nModel accuracy on test data: {test_acc * 100:.2f}%')

# 5. Predict and show results
def predict_and_show(model, test_images, test_labels):
    predictions = model.predict(test_images)

    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        predicted_label = tf.argmax(predictions[i]).numpy()
        true_label = test_labels[i]
        plt.xlabel(f'Pred: {predicted_label}, True: {true_label}')
    plt.show()

# Execute project steps
if __name__ == "__main__":
    # 1. Load data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    # 2. Create model
    model = create_model()

    # 3. Compile and train the model
    compile_and_train_model(model, train_images, train_labels)

    # 4. Evaluate the model
    evaluate_model(model, test_images, test_labels)

    # 5. Predict and show results
    predict_and_show(model, test_images, test_labels)
