import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=67, stratify=iris.target
)

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

tsne = TSNE(n_components=2, random_state=67, perplexity=30)
X_tsne = tsne.fit_transform(iris.data)

test_indices = []
for val in X_test:
    for i, orig in enumerate(iris.data):
        if (val == orig).all(): 
            test_indices.append(i)
            break

colors = ['blue', 'green', 'red']

print(f"{'k':<5} | {'Accuracy':<10}")
print("-" * 20)
for k in range(1, 11):
    knn_temp = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
    acc = accuracy_score(y_test, knn_temp.predict(X_test_s))
    print(f"{k:<5} | {acc:.4f}")

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

print("\nMetryki klasyfikacji (k=5):\n", classification_report(y_test, y_pred, target_names=iris.target_names))

plt.figure(figsize=(10, 7))
for i, name in enumerate(iris.target_names):
    idx = iris.target == i
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=colors[i], label=name, alpha=0.7)
plt.title("Wizualizacja t-SNE: Zbiór oryginalny")
plt.legend()
plt.savefig("wizualizacja.png")
plt.close()

plt.figure(figsize=(12, 8))
for i in range(3):
    idx = iris.target == i
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=colors[i], alpha=0.1, label=iris.target_names[i])

for i, idx_in_tsne in enumerate(test_indices):
    true_label = y_test[i]
    pred_label = y_pred[i]
    
    if true_label == pred_label:
        plt.scatter(X_tsne[idx_in_tsne, 0], X_tsne[idx_in_tsne, 1], 
                    c=colors[true_label], marker='o', edgecolors='black', s=80)
    else:
        plt.scatter(X_tsne[idx_in_tsne, 0], X_tsne[idx_in_tsne, 1], 
                    c='black', marker='x', s=150, linewidths=2)
        
        info_text = f"Predykcja: {iris.target_names[pred_label]}\nPrawda: {iris.target_names[true_label]}"
        plt.annotate(info_text, (X_tsne[idx_in_tsne, 0], X_tsne[idx_in_tsne, 1]), 
                     textcoords="offset points", xytext=(40,-5), ha='center', 
                     fontsize=7, color='black',
                     bbox=dict(boxstyle='round,pad=0.2', alpha=0.1))

plt.title("Wizualizacja t-SNE: Predykcja modelu KNN (k=5)")
plt.grid(True, alpha=0.2)
plt.legend(title="Gatunki")
plt.savefig("wizualizacja_predykcja.png")
