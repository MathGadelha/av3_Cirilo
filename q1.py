import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funções de ativação e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Função de ativação do Perceptron
def sign(u):
    return 1 if u >= 0 else -1

# Função para calcular o EQM (Erro Quadrático Médio) no ADALINE
def EQM(X, Y, w):
    p_1, N = X.shape
    eq = 0
    for t in range(N):
        x_t = X[:, t].reshape(p_1, 1)
        u_t = w.T @ x_t
        d_t = Y[0, t]
        eq += (d_t - u_t[0, 0]) ** 2
    return eq / (2 * N)

# Leitura dos dados
data = pd.read_csv('spiral.csv', header=None)
X = data.iloc[:, :2].values
Y = data.iloc[:, 2].values

# Visualização inicial dos dados
plt.figure(figsize=(8, 6))
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=90, marker='*', color='blue', label='Classe +1')
plt.scatter(X[Y == -1, 0], X[Y == -1, 1], s=90, marker='s', color='red', label='Classe -1')
plt.title("Visualização Inicial dos Dados")
plt.legend()
plt.show()

# Preparação dos dados
X = X.T
Y = Y.reshape(1, -1)
p, N = X.shape
X = np.vstack((-np.ones((1, N)), X))

# Funções de treinamento para Perceptron e ADALINE
def train_perceptron(X, Y, lr=0.01, max_epochs=1000):
    w = np.random.random_sample((X.shape[0], 1)) - 0.5
    for epoch in range(max_epochs):
        erro = False
        for t in range(N):
            x_t = X[:, t].reshape(p + 1, 1)
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = float(Y[0, t])
            e_t = d_t - y_t
            if y_t != d_t:
                w += (lr * e_t * x_t) / 2
                erro = True
        if not erro:
            print(f"Perceptron convergiu em {epoch} épocas.")
            break
    return w

def train_adaline(X, Y, lr=0.01, max_epochs=1000, pr=1e-5):
    w = np.random.random_sample((X.shape[0], 1)) - 0.5
    EQM1, EQM2 = 1, 0
    hist = []
    for epoch in range(max_epochs):
        EQM1 = EQM(X, Y, w)
        hist.append(EQM1)
        if abs(EQM1 - EQM2) <= pr:
            print(f"ADALINE convergiu em {epoch} épocas.")
            break
        for t in range(N):
            x_t = X[:, t].reshape(p + 1, 1)
            u_t = w.T @ x_t
            d_t = Y[0, t]
            e_t = d_t - u_t
            w += lr * e_t * x_t
        EQM2 = EQM(X, Y, w)
    return w, hist

# Treinamento do Perceptron
w_perceptron = train_perceptron(X, Y)
x_axis = np.linspace(X[1, :].min() - 1, X[1, :].max() + 1, 100)
x2_perceptron = -w_perceptron[1, 0] / w_perceptron[2, 0] * x_axis + w_perceptron[0, 0] / w_perceptron[2, 0]

# Visualização do Perceptron
plt.figure(figsize=(8, 6))
plt.scatter(X[1, Y[0, :] == 1], X[2, Y[0, :] == 1], s=90, marker='*', color='blue', label='Classe +1')
plt.scatter(X[1, Y[0, :] == -1], X[2, Y[0, :] == -1], s=90, marker='s', color='red', label='Classe -1')
plt.plot(x_axis, x2_perceptron, color='green', label='Reta de Decisão (Perceptron)')
plt.legend()
plt.title("Perceptron - Reta de Decisão")
plt.show()

# Treinamento do ADALINE
w_adaline, hist_adaline = train_adaline(X, Y)
x2_adaline = -w_adaline[1, 0] / w_adaline[2, 0] * x_axis + w_adaline[0, 0] / w_adaline[2, 0]

# Visualização do ADALINE
plt.figure(figsize=(8, 6))
plt.scatter(X[1, Y[0, :] == 1], X[2, Y[0, :] == 1], s=90, marker='*', color='blue', label='Classe +1')
plt.scatter(X[1, Y[0, :] == -1], X[2, Y[0, :] == -1], s=90, marker='s', color='red', label='Classe -1')
plt.plot(x_axis, x2_adaline, color='orange', label='Reta de Decisão (ADALINE)')
plt.legend()
plt.title("ADALINE - Reta de Decisão")
plt.show()

# Curva de aprendizado do ADALINE
plt.figure(figsize=(8, 6))
plt.plot(hist_adaline, color='blue', linewidth=2, label="EQM x Épocas")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.title("Curva de Aprendizado - ADALINE")
plt.legend()
plt.grid()
plt.show()

# Implementação da MLP
class MLP:
    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(n_in, n_out) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((1, n_out)) for n_out in layer_sizes[1:]]
        self.activation_function = relu if activation == 'relu' else sigmoid
        self.activation_derivative = relu_derivative if activation == 'relu' else sigmoid_derivative

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            a = self.activation_function(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, Y, learning_rate):
        m = X.shape[0]
        dz = self.a[-1] - Y.reshape(-1, 1)
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.activation_derivative(self.a[i])
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, Y, epochs=1000, learning_rate=0.1, print_loss=False):
        loss_history = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - Y.reshape(-1, 1)) ** 2)
            loss_history.append(loss)
            self.backward(X, Y, learning_rate)
            if print_loss and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss_history

# Função para calcular a acurácia
def calculate_accuracy(model, X, Y):
    correct_predictions = 0
    total_predictions = Y.shape[1]
    
    for t in range(total_predictions):
        x_t = X[:, t].reshape(-1, 1)
        if model == 'perceptron':
            u_t = w_perceptron.T @ x_t
            prediction = sign(u_t[0, 0])
        elif model == 'adaline':
            u_t = w_adaline.T @ x_t
            prediction = sign(u_t[0, 0])
        else:  # MLP
            prediction = 1 if mlp.forward(x_t[1:].T) >= 0.5 else -1
        
        if prediction == Y[0, t]:
            correct_predictions += 1
    
    return correct_predictions / total_predictions * 100

# Acurácia do Perceptron
accuracy_perceptron = calculate_accuracy('perceptron', X, Y)
print(f"Acurácia do Perceptron: {accuracy_perceptron:.2f}%")

# Acurácia do ADALINE
accuracy_adaline = calculate_accuracy('adaline', X, Y)
print(f"Acurácia do ADALINE: {accuracy_adaline:.2f}%")

# Treinamento da MLP
layer_sizes = [2, 5, 1]  # Camada de entrada, 1 camada oculta com 5 neurônios, 1 saída
mlp = MLP(layer_sizes, activation='sigmoid')

# Dados de entrada para MLP
X_mlp = X[1:].T  # Excluímos o bias (-1)
Y_mlp = (Y.T + 1) / 2  # Convertendo para [0, 1] para saída da MLP

loss_history = mlp.train(X_mlp, Y_mlp, epochs=1000, learning_rate=0.01, print_loss=True)

# Acurácia da MLP
accuracy_mlp = calculate_accuracy('mlp', X, Y)
print(f"Acurácia da MLP: {accuracy_mlp:.2f}%")

# Função para calcular estatísticas
def calculate_statistics(accuracies):
    mean = np.mean(accuracies)
    std_dev = np.std(accuracies)
    max_value = np.max(accuracies)
    min_value = np.min(accuracies)
    return mean, std_dev, max_value, min_value

# Simular múltiplos treinamentos e calcular acurácias
def simulate_accuracies(model, X, Y, num_simulations=10):
    accuracies = []
    for _ in range(num_simulations):
        if model == 'perceptron':
            global w_perceptron
            w_perceptron = train_perceptron(X, Y)
        elif model == 'adaline':
            global w_adaline
            w_adaline, _ = train_adaline(X, Y)
        else:  # MLP
            layer_sizes = [2, 5, 1]
            global mlp
            mlp = MLP(layer_sizes, activation='sigmoid')
            X_mlp = X[1:].T
            Y_mlp = (Y.T + 1) / 2
            mlp.train(X_mlp, Y_mlp, epochs=1000, learning_rate=0.01)
        
        accuracy = calculate_accuracy(model, X, Y)
        accuracies.append(accuracy)
    return accuracies

# Coletar acurácias para cada modelo
num_simulations = 10
accuracies_perceptron = simulate_accuracies('perceptron', X, Y, num_simulations)
accuracies_adaline = simulate_accuracies('adaline', X, Y, num_simulations)
accuracies_mlp = simulate_accuracies('mlp', X, Y, num_simulations)

# Calcular estatísticas para cada modelo
stats_perceptron = calculate_statistics(accuracies_perceptron)
stats_adaline = calculate_statistics(accuracies_adaline)
stats_mlp = calculate_statistics(accuracies_mlp)

# Criar tabela formatada
table_header = f"{'Modelo':<15}{'Média':<10}{'Desvio Padrão':<15}{'Maior Valor':<15}{'Menor Valor':<15}"
table_separator = "-" * len(table_header)
table_content = (
    f"{'Perceptron':<15}{stats_perceptron[0]:<10.2f}{stats_perceptron[1]:<15.2f}{stats_perceptron[2]:<15.2f}{stats_perceptron[3]:<15.2f}\n"
    f"{'ADALINE':<15}{stats_adaline[0]:<10.2f}{stats_adaline[1]:<15.2f}{stats_adaline[2]:<15.2f}{stats_adaline[3]:<15.2f}\n"
    f"{'MLP':<15}{stats_mlp[0]:<10.2f}{stats_mlp[1]:<15.2f}{stats_mlp[2]:<15.2f}{stats_mlp[3]:<15.2f}"
)

# Exibir tabela
print(table_header)
print(table_separator)
print(table_content)
