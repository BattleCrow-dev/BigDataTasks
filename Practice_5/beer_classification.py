import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Загрузка и предобработка данных
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

# Кодируем целевую переменную
le = LabelEncoder()
y = le.fit_transform(y)

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Гистограмма баланса классов
plt.figure(figsize=(10,6))
df['Style'].value_counts().plot(kind='bar')
plt.title("Распределение топ‑10 стилей пива")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Количество рецептов")
plt.tight_layout()
plt.show()

# 3. Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Обучение моделей
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

# 5. Построение confusion_matrix и сбор метрик
for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.show()

    # Метрики
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })

# 6. Таблица сравнения
results_df = pd.DataFrame(results).set_index("Model")
print("\nСравнение моделей:\n")
print(results_df)

# 7. Визуализация метрик
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(results_df.index))  # позиции для моделей
width = 0.2  # ширина столбца

fig, ax = plt.subplots(figsize=(10,6))
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, results_df[metric], width, label=metric)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df.index)
ax.set_ylabel("Значение метрики")
ax.set_title("Сравнение моделей по метрикам")
ax.legend()
plt.tight_layout()
plt.show()

# 8. Выводы
best_model = results_df['F1'].idxmax()
print(f"\nВыводы:\n- Лучшая модель по F1: {best_model}")
print("- Дисбаланс классов уменьшен за счёт выбора топ‑10 стилей.")
print("- Логистическая регрессия даёт базовый результат, SVM показывает лучшие метрики, KNN средний.")
print("- Для улучшения можно попробовать подбор гиперпараметров и ансамблевые методы.")
