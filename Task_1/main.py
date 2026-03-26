import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.describe())

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target, 
    test_size=0.2, 
    random_state=67,
    stratify=iris.target
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n", classification_report(y_test, y_pred, target_names=iris.target_names))

tsne = TSNE(n_components=2, random_state=67)
X_tsne = tsne.fit_transform(iris.data)

target_names_list = [iris.target_names[i] for i in iris.target]

colors = {
    'setosa': 'red',
    'versicolor': 'green',
    'virginica': 'blue'
}

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=X_tsne[:, 0], 
    y=X_tsne[:, 1], 
    hue=target_names_list, 
    palette=colors,
    alpha=0.8
)

plt.title("Wizualizacja zbioru Iris")
plt.xlabel("wymiar 1")
plt.ylabel("wymiar 2")
plt.legend(title="Gatunki kwiatów")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("wizualizacja.png")