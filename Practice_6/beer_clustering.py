import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# ------------------------------
# 1. Загрузка и предобработка данных
# ------------------------------
print("Загрузка и предобработка данных...")
df = pd.read_csv('recipeData.csv', encoding='latin1')

# Удаляем строки с пропусками в ключевых признаках
df = df.dropna(subset=['Style', 'ABV', 'IBU', 'Color', 'BoilGravity'])

# Берём только топ‑10 стилей
top_styles = df['Style'].value_counts().head(10).index
df = df[df['Style'].isin(top_styles)]

# Признаки и целевая переменная
X = df[['ABV', 'IBU', 'Color', 'BoilGravity']]
y = df['Style']

# Кодируем целевую переменную (для анализа, не для кластеризации)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Размерность данных:", X_scaled.shape)

# Чтобы ускорить работу, берём подвыборку (например, 500 точек)
sample_size = min(500, len(X_scaled))
X_sample = X_scaled[np.random.choice(len(X_scaled), sample_size, replace=False)]

# ------------------------------
# 2. K-Means: подбор числа кластеров
# ------------------------------
inertia, silhouette = [], []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_sample)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_sample, labels))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K, inertia, 'bx-')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Правило локтя")

plt.subplot(1,2,2)
plt.plot(K, silhouette, 'ro-')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Коэффициент силуэта")
plt.show()

optimal_k = K[np.argmax(silhouette)]
print("Оптимальное количество кластеров (по силуэту):", optimal_k)

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels_kmeans = kmeans.fit_predict(X_sample)

# ------------------------------
# 3. Иерархическая кластеризация (дендрограмма с пояснениями)
# ------------------------------
plt.figure(figsize=(12,6))
Z = sch.linkage(X_sample, method='ward')
sch.dendrogram(
    Z,
    truncate_mode='lastp',  # показываем только последние p кластеров
    p=20,
    leaf_rotation=45.,
    leaf_font_size=10,
    show_contracted=True,
    color_threshold=40
)
plt.title("Усечённая дендрограмма (20 кластеров)")
plt.xlabel("Кластеры (объединённые группы точек)")
plt.ylabel("Евклидово расстояние (мера различия)")
plt.show()

hc = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
labels_hc = hc.fit_predict(X_sample)

# ------------------------------
# 4. DBSCAN
# ------------------------------
db = DBSCAN(eps=1.0, min_samples=8)
labels_db = db.fit_predict(X_sample)

print("Кластеры DBSCAN:", len(set(labels_db)) - (1 if -1 in labels_db else 0))
print("Выбросы (label=-1):", list(labels_db).count(-1))

# ------------------------------
# 5. Визуализация кластеров (3D графики)
# ------------------------------

# Сравнение всех трёх алгоритмов в 3D
fig = plt.figure(figsize=(18,5))

ax = fig.add_subplot(131, projection='3d')
ax.scatter(X_sample[:,0], X_sample[:,1], X_sample[:,2], c=labels_kmeans, cmap="viridis")
ax.set_title("K-Means (3D)")
ax.set_xlabel("ABV"); ax.set_ylabel("IBU"); ax.set_zlabel("Color")

ax = fig.add_subplot(132, projection='3d')
ax.scatter(X_sample[:,0], X_sample[:,1], X_sample[:,2], c=labels_hc, cmap="plasma")
ax.set_title("Иерархическая (3D)")
ax.set_xlabel("ABV"); ax.set_ylabel("IBU"); ax.set_zlabel("Color")

ax = fig.add_subplot(133, projection='3d')
ax.scatter(X_sample[:,0], X_sample[:,1], X_sample[:,2], c=labels_db, cmap="tab10")
ax.set_title("DBSCAN (3D)")
ax.set_xlabel("ABV"); ax.set_ylabel("IBU"); ax.set_zlabel("Color")

plt.tight_layout()
plt.show()

# Barplot профилей кластеров (K-Means)
df_kmeans = pd.DataFrame(X_sample, columns=['ABV','IBU','Color','BoilGravity'])
df_kmeans['Cluster'] = labels_kmeans
cluster_means = df_kmeans.groupby("Cluster").mean()

cluster_means.plot(kind="bar", figsize=(12,6))
plt.title("Средние значения признаков по кластерам (K-Means)")
plt.ylabel("Нормализованные значения")
plt.xticks(rotation=0)
plt.show()

# ------------------------------
# 6. Сравнение силуэт-метрик
# ------------------------------
print("Silhouette K-Means:", silhouette_score(X_sample, labels_kmeans))
print("Silhouette HC:", silhouette_score(X_sample, labels_hc))
if len(set(labels_db)) > 1:
    print("Silhouette DBSCAN:", silhouette_score(X_sample, labels_db))
else:
    print("Silhouette DBSCAN: невозможно вычислить (все точки в одном кластере)")
