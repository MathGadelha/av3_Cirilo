import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para os dados
recfac_path = "RecFac"

# Dimensões para redimensionamento
data = []
labels = []
resize_dims = (50, 50)  # Escolhido 50x50

# Leitura e pré-processamento das imagens
for label, subdir in enumerate(os.listdir(recfac_path)):
    person_path = os.path.join(recfac_path, subdir)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, resize_dims)
                data.append(resized_image.flatten())
                labels.append(label)

# Convertendo os dados para numpy arrays
data = np.array(data) / 255.0  # Normalização
labels = np.array(labels)

# Função para divisão dos dados
def split_data(data, labels, train_ratio=0.8):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_size = int(train_ratio * len(data))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]

# Função para calcular acurácia
def calculate_metrics(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, model_name, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')
    plt.show()

# Classes de modelos
class Perceptron:
    def _init_(self, input_dim, num_classes, learning_rate=0.01, max_epochs=50):
        self.weights = np.random.randn(input_dim, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            for i in range(X.shape[0]):
                x_t = X[i].reshape(1, -1)
                y_t = np.zeros((1, self.weights.shape[1]))
                y_t[0, y[i]] = 1

                u_t = np.dot(x_t, self.weights) + self.bias
                y_pred = np.argmax(u_t)
                if y_pred != y[i]:
                    self.weights[:, y[i]] += self.learning_rate * x_t.flatten()
                    self.weights[:, y_pred] -= self.learning_rate * x_t.flatten()

    def predict(self, X):
        return np.argmax(np.dot(X, self.weights) + self.bias, axis=1)

class Adaline:
    def _init_(self, input_dim, num_classes, learning_rate=0.001, max_epochs=50):
        self.weights = np.random.randn(input_dim, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            u = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(u)
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(len(y)), y] = 1
            error = y_one_hot - y_pred
            self.weights += self.learning_rate * np.dot(X.T, error)
            self.bias += self.learning_rate * error.mean(axis=0)

    def predict(self, X):
        u = np.dot(X, self.weights) + self.bias
        return np.argmax(u, axis=1)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class MLP:
    def _init_(self, input_dim, hidden_dim, num_classes, learning_rate=0.001, max_epochs=50):
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.01
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, num_classes) * 0.01
        self.bias_output = np.zeros((1, num_classes))
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.relu(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = self.softmax(final_input)

            y_one_hot = np.zeros_like(final_output)
            y_one_hot[np.arange(len(y)), y] = 1

            error_output = y_one_hot - final_output
            error_hidden = np.dot(error_output, self.weights_hidden_output.T) * self.relu_derivative(hidden_output)

            self.weights_hidden_output += self.learning_rate * np.dot(hidden_output.T, error_output)
            self.bias_output += self.learning_rate * error_output.mean(axis=0)

            self.weights_input_hidden += self.learning_rate * np.dot(X.T, error_hidden)
            self.bias_hidden += self.learning_rate * error_hidden.mean(axis=0)

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        return np.argmax(final_input, axis=1)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Divisão dos dados
data_train, data_test, labels_train, labels_test = split_data(data, labels)
num_classes = len(np.unique(labels))

# Avaliação dos modelos
models = {
    "Perceptron": Perceptron(input_dim=data_train.shape[1], num_classes=num_classes),
    "Adaline": Adaline(input_dim=data_train.shape[1], num_classes=num_classes),
    "MLP": MLP(input_dim=data_train.shape[1], hidden_dim=64, num_classes=num_classes)
}

for model_name, model in models.items():
    model.train(data_train, labels_train)
    predictions = model.predict(data_test)
    accuracy = calculate_metrics(labels_test, predictions)
    print(f"Acurácia do {model_name}: {accuracy:.2f}")
    plot_confusion_matrix(labels_test, predictions, model_name, num_classes)