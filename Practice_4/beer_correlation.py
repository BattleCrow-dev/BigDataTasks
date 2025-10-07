import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Загрузка и предобработка ===
df = pd.read_csv('recipeData.csv', encoding='latin1')

# Берём только числовые признаки
num_df = df.select_dtypes(include=[np.number])

# Заполняем пропуски средними
num_df = num_df.fillna(num_df.mean())

# Удаление выбросов по IBU и низких значений
Q1 = num_df["IBU"].quantile(0.25)
Q3 = num_df["IBU"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# фильтруем: убираем выбросы и значения IBU < 5
clean_df = num_df[(num_df["IBU"] >= lower_bound) &
                  (num_df["IBU"] <= upper_bound) &
                  (num_df["IBU"] >= 1)]

print(f"Размер выборки до очистки: {len(num_df)}")
print(f"Размер выборки после очистки: {len(clean_df)}")

# === 2. Корреляционная матрица ===
plt.figure(figsize=(10, 6))
sns.heatmap(clean_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Корреляционная матрица (после очистки)")
plt.show()

# === 3. Линейная регрессия ABV ~ IBU ===
X = clean_df["IBU"].values
y = clean_df["ABV"].values

X_mean, y_mean = X.mean(), y.mean()

# Наклон и сдвиг
b1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
b0 = y_mean - b1 * X_mean

print(f"Наклон (slope): {b1:.3f}")
print(f"Сдвиг (intercept): {b0:.3f}")

# Предсказания
y_pred = b0 + b1 * X

# MSE
mse = np.mean((y - y_pred)**2)
print(f"MSE: {mse:.3f}")

# === 4. Визуализация регрессии ===
plt.figure(figsize=(8,6))
plt.scatter(X, y, alpha=0.5, label="Данные (очищенные)")
plt.plot(X, y_pred, color="red", label="Линия регрессии")
plt.xlabel("IBU (горечь)")
plt.ylabel("ABV (крепость, %)")
plt.title("Регрессия ABV ~ IBU (без выбросов и низких значений)")
plt.legend()
plt.show()
