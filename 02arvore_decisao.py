import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

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

# Implementação da Árvore de Decisão com Gini
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split  
        self.tree = None

    def _gini_index(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def _split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def _get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self._split(index, row[index], dataset)
                gini = self._gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _build_tree(self, train, depth):
        
        if len(train) <= self.min_samples_leaf or len(train) < self.min_samples_split:
            return max(set([row[-1] for row in train]), key=[row[-1] for row in train].count) if len(train) > 0 else 0
        if depth >= self.max_depth:
            return max(set([row[-1] for row in train]), key=[row[-1] for row in train].count)
        node = self._get_split(train)
        left, right = node['groups']
        del (node['groups'])
        node['left'] = self._build_tree(left, depth + 1)
        node['right'] = self._build_tree(right, depth + 1)
        return node

    def fit(self, X, y):
        dataset = np.c_[X, y]
        self.tree = self._build_tree(dataset.tolist(), 0)

    def _predict_row(self, row, tree):
        if isinstance(tree, dict):
            if row[tree['index']] < tree['value']:
                return self._predict_row(row, tree['left'])
            else:
                return self._predict_row(row, tree['right'])
        else:
            return tree

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _predict_proba_row(self, row, tree):
        if isinstance(tree, dict):
            if row[tree['index']] < tree['value']:
                return self._predict_proba_row(row, tree['left'])
            else:
                return self._predict_proba_row(row, tree['right'])
        else:
            return {0: 1 - tree, 1: tree} if isinstance(tree, (float, int)) else {0: 0.5, 1: 0.5}

    def predict_proba(self, X):
        return np.array([[proba[0], proba[1]] for proba in (self._predict_proba_row(row, self.tree) for row in X)])


data = pd.read_csv("bostonbin.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

def _predict_proba_row(self, row, tree):
    if isinstance(tree, dict):
        if row[tree['index']] < tree['value']:
            return self._predict_proba_row(row, tree['left'])
        else:
            return self._predict_proba_row(row, tree['right'])
    else:
        if isinstance(tree, list):
            total = len(tree)
            count_0 = sum(1 for x in tree if x == 0)
            count_1 = sum(1 for x in tree if x == 1)
            return {0: count_0 / total, 1: count_1 / total}
        return {0: 1 - tree, 1: tree}

def predict_proba(self, X):
    return np.array([self._predict_proba_row(row, self.tree) for row in X])


# Divisão dos dados
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalizando os dados
X_train, X_test = normalizar_dados(X_train, X_test)

# Ajuste de hiperparâmetros e Grid Search para Árvore de Decisão
max_depth_values = range(1, 11)
min_samples_leaf_values = range(1, 6)
best_arvore_model = None
best_arvore_score = -np.inf
for max_depth in max_depth_values:
    for min_samples_leaf in min_samples_leaf_values:
        arvore_model = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        arvore_model.fit(X_train, y_train)
        y_pred_arvore = arvore_model.predict(X_test)
        acuracia_arvore, _, _, _ = calcular_metricas(y_test, y_pred_arvore)
        if acuracia_arvore > best_arvore_score:
            best_arvore_score = acuracia_arvore
            best_arvore_model = arvore_model

# Exibição dos melhores resultados para Árvore de Decisão
print("Melhor modelo Árvore de Decisão:")
print(f"Max Depth: {best_arvore_model.max_depth}, Min Samples Leaf: {best_arvore_model.min_samples_leaf}")
y_pred_arvore = best_arvore_model.predict(X_test)
acuracia_arvore, _, _, _ = calcular_metricas(y_test, y_pred_arvore)
print(f"Acurácia: {acuracia_arvore:.4f}")

import os

# Previsão probabilística
y_proba = best_arvore_model.predict_proba(X_test)
y_scores = y_proba[:, 1]  

output_dir = "imagens"
os.makedirs(output_dir, exist_ok=True)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Árvore de Decisão - Curva ROC')
plt.legend(loc='lower right')
roc_path = os.path.join(output_dir, "arvore_decisao - curva_roc.png")
plt.savefig(roc_path, dpi=300)
plt.show()

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='green', label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Árvore de Decisão - Curva Precision-Recall')
plt.legend(loc='lower left')
pr_path = os.path.join(output_dir, "arvore_decisao - curva_precision_recall.png")
plt.savefig(pr_path, dpi=300)
plt.show()

