import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# Funções auxiliares
def normalizar_dados(X_train, X_test):
    """
    Normaliza os dados com base no mínimo e máximo do conjunto de treino.

    Parâmetros:
        X_train (ndarray): Dados de treino.
        X_test (ndarray): Dados de teste.

    Retorna:
        X_train_norm (ndarray): Dados de treino normalizados.
        X_test_norm (ndarray): Dados de teste normalizados com base no treino.
    """
    # Calcula o mínimo e o máximo do conjunto de treino
    X_min = np.min(X_train, axis=0)
    X_max = np.max(X_train, axis=0)

    # Normaliza o conjunto de treino
    X_train_norm = (X_train - X_min) / (X_max - X_min)

    # Normaliza o conjunto de teste com base nos valores do treino
    X_test_norm = (X_test - X_min) / (X_max - X_min)

    return X_train_norm, X_test_norm


def calcular_metricas(y_true, y_pred):
    """Calcula acurácia, precisão, revocação e F1-Score manualmente."""
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    acuracia = (tp + tn) / len(y_true)
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    revocacao = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precisao * revocacao) / (precisao + revocacao) if (precisao + revocacao) > 0 else 0

    return acuracia, precisao, revocacao, f1_score


# Implementação do Kernel RBF para SVM
def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


# Implementação da SVM com Kernel RBF
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iter=1000, C=1.0, gamma=0.01):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.C = C
        self.gamma = gamma
        self.alpha = None
        self.bias = 0
        self.X_train = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X_train = X
        self.alpha = np.zeros(n_samples)
        kernel_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = rbf_kernel(X[i], X[j], self.gamma)

        for _ in range(self.n_iter):
            alpha_prev = np.copy(self.alpha)
            for i in range(n_samples):
                condition = y[i] * (np.sum(self.alpha * y * kernel_matrix[i]) - self.bias) >= 1
                if condition:
                    self.alpha[i] -= self.learning_rate * (2 * self.lambda_param * self.alpha[i])
                else:
                    self.alpha[i] -= self.learning_rate * (
                            2 * self.lambda_param * self.alpha[i] - np.dot(kernel_matrix[i], y))
                    self.bias -= self.learning_rate * y[i]

            # Verificação de convergência
            if np.linalg.norm(self.alpha - alpha_prev) < 1e-5:
                break

    def predict(self, X):
        n_samples = X.shape[0]
        kernel_matrix = np.zeros((n_samples, len(self.X_train)))

        for i in range(n_samples):
            for j in range(len(self.X_train)):
                kernel_matrix[i, j] = rbf_kernel(X[i], self.X_train[j], self.gamma)

        return np.sign(np.sum(self.alpha * kernel_matrix, axis=1) - self.bias)


# Leitura do conjunto de dados
data = pd.read_csv("bostonbin.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Divisão dos dados
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalizando os dados
X_train, X_test = normalizar_dados(X_train, X_test)

# Ajuste de hiperparâmetros e Grid Search para SVM
C_values = [2 ** i for i in range(-5, 16, 2)]
gamma_values = [2 ** i for i in range(-15, 4, 2)]
best_svm_model = None
best_svm_score = -np.inf
results = []

for C in C_values:
    for gamma in gamma_values:
        svm_model = SVM(C=C, gamma=gamma)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        acuracia_svm, _, _, _ = calcular_metricas(y_test, y_pred_svm)
        results.append((C, gamma, acuracia_svm))
        if acuracia_svm > best_svm_score:
            best_svm_score = acuracia_svm
            best_svm_model = svm_model

# Exibição dos melhores resultados para SVM
print("Melhor modelo SVM:")
print(f"C: {best_svm_model.C}, Gamma: {best_svm_model.gamma}")
y_pred_svm = best_svm_model.predict(X_test)
acuracia_svm, precisao_svm, revocacao_svm, f1_svm = calcular_metricas(y_test, y_pred_svm)
print(f"Acurácia: {acuracia_svm:.4f}")
print(f"Precisão: {precisao_svm:.4f}")
print(f"Revocação: {revocacao_svm:.4f}")
print(f"F1-Score: {f1_svm:.4f}")

# Plotando os resultados
results = np.array(results)
C_values_plot = np.unique(results[:, 0])
gamma_values_plot = np.unique(results[:, 1])
accuracy_matrix = results[:, 2].reshape(len(C_values_plot), len(gamma_values_plot))

plt.figure(figsize=(10, 8))
plt.imshow(accuracy_matrix, interpolation='nearest', cmap='viridis', origin='lower',
           extent=[min(gamma_values_plot), max(gamma_values_plot), min(C_values_plot), max(C_values_plot)])
plt.colorbar(label='Acurácia')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title('Acurácia da SVM para diferentes valores de C e Gamma')
plt.savefig("imagens/acuracia_svm.png")
plt.show()

# Plotando a curva ROC
y_scores = y_pred_svm
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"Curva ROC (área = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.savefig("imagens/curva_roc.png")
plt.show()

# Plotando a curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Revocação')
plt.ylabel('Precisão')
plt.title('Curva Precision-Recall')
plt.savefig("imagens/curva_precision_recall.png")
plt.show()
