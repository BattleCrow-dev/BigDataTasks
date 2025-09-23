import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Загрузка данных
df = pd.read_csv("insurance.csv")

# 2. Описательная статистика
print("Описательная статистика по числовым признакам:")
print(df.describe())

# 3. Гистограммы для числовых признаков
num_cols = ["age", "bmi", "children", "charges"]
df[num_cols].hist(bins=30, figsize=(12, 8))
plt.suptitle("Гистограммы числовых признаков")
plt.show()

# 4. Центральные меры и разброс для bmi и charges
for col in ["bmi", "charges"]:
    mean_val = df[col].mean()
    median_val = df[col].median()
    mode_val = df[col].mode()[0]
    std_val = df[col].std()
    var_val = df[col].var()

    print(f"\nПоказатели для {col}:")
    print(f"Среднее: {mean_val:.2f}")
    print(f"Медиана: {median_val:.2f}")
    print(f"Мода: {mode_val:.2f}")
    print(f"Стандартное отклонение: {std_val:.2f}")
    print(f"Дисперсия: {var_val:.2f}")

    # Гистограмма с тремя вертикальными линиями
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=False, color="skyblue")
    plt.axvline(mean_val, color="red", linestyle="--", label="Среднее")
    plt.axvline(median_val, color="green", linestyle="--", label="Медиана")
    plt.axvline(mode_val, color="blue", linestyle="--", label="Мода")
    plt.legend()
    plt.title(f"Гистограмма {col} с мерами центральной тенденции")
    plt.show()

# 5. Box-plot для числовых признаков
plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.suptitle("Box-plot для числовых признаков")
plt.show()

# 6. Проверка ЦПТ на примере charges
sample_sizes = [30, 50, 100]
for n in sample_sizes:
    sample_means = []
    for _ in range(300):
        sample = df["charges"].sample(n, replace=True)
        sample_means.append(sample.mean())
    plt.hist(sample_means, bins=30, alpha=0.7, color="orange")
    plt.title(f"Распределение средних charges (n={n})")
    plt.xlabel("Среднее значение")
    plt.ylabel("Частота")
    plt.show()
    print(f"n={n}: среднее={np.mean(sample_means):.2f}, std={np.std(sample_means):.2f}")

# 7. Доверительные интервалы для charges и bmi
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean - h, mean + h

for col in ["charges", "bmi"]:
    ci95 = confidence_interval(df[col], 0.95)
    ci99 = confidence_interval(df[col], 0.99)
    print(f"\nДоверительные интервалы для {col}:")
    print(f"95%: {ci95}")
    print(f"99%: {ci99}")

# 8. Проверка нормальности (KS-тест и Q-Q plot)
for col in ["bmi", "charges"]:
    mu, sigma = df[col].mean(), df[col].std()
    # KS-тест
    ks_stat, p_val = stats.kstest(df[col], 'norm', args=(mu, sigma))
    print(f"\nПроверка нормальности для {col}:")
    print(f"KS-статистика={ks_stat:.4f}, p-value={p_val:.4f}")

    # Q-Q plot
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(f"Q-Q plot для {col}")
    plt.show()
