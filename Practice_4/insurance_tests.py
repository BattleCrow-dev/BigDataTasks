import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

# === 1. Загрузка и предобработка ===
df = pd.read_csv("insurance.csv")

print("Первые строки датафрейма:")
print(df.head())

print("\nПроверка пропусков:")
print(df.isnull().sum())

print("\nУникальные регионы:")
print(df['region'].unique())

# === 2. Однофакторный ANOVA: влияние региона на BMI ===
# Способ 1: scipy.stats.f_oneway
regions = df['region'].unique()
bmi_groups = [df[df['region'] == region]['bmi'] for region in regions]
f_stat, p_val = f_oneway(*bmi_groups)
print(f"\nANOVA (scipy): F={f_stat:.3f}, p={p_val:.6f}")

# Способ 2: statsmodels + anova_lm
model1 = ols('bmi ~ C(region)', data=df).fit()
anova1 = anova_lm(model1)
print("\nANOVA (statsmodels):")
print(anova1)

# Визуализация распределений BMI по регионам
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='region', y='bmi', order=sorted(df['region'].unique()))
sns.stripplot(data=df, x='region', y='bmi', order=sorted(df['region'].unique()),
              color='black', alpha=0.35, jitter=0.2)
plt.title("Распределение BMI по регионам")
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

# === 3. Парные t-тесты + поправка Бонферрони ===
pairs = list(itertools.combinations(regions, 2))
alpha = 0.05
alpha_bonf = alpha / len(pairs)
print(f"\nПарные t-тесты (alpha={alpha}, Bonferroni alpha={alpha_bonf:.5f}):")

results_pairs = []
for r1, r2 in pairs:
    g1 = df[df['region'] == r1]['bmi']
    g2 = df[df['region'] == r2]['bmi']
    t_stat, p = ttest_ind(g1, g2, equal_var=False)
    p_adj = p * len(pairs)
    results_pairs.append((r1, r2, t_stat, p, p_adj))
    print(f"{r1} vs {r2}: t={t_stat:.3f}, p={p:.6f}, p(Bonf)={p_adj:.6f}, "
          f"{'значимо' if p_adj < alpha else 'не значимо'}")

# === 4. Пост-хок тест Тьюки (один фактор: регион) ===
tukey1 = pairwise_tukeyhsd(endog=df['bmi'], groups=df['region'], alpha=0.05)
print("\nТест Тьюки (регион):")
print(tukey1)

# График доверительных интервалов различий (Тьюки, регион)
fig1 = tukey1.plot_simultaneous(figsize=(8,6))
plt.title("Пост-хок Тьюки: различия BMI между регионами")
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

# === 5. Двухфакторный ANOVA: регион + пол ===
model2 = ols('bmi ~ C(region) + C(sex) + C(region):C(sex)', data=df).fit()
anova2 = anova_lm(model2)
print("\nДвухфакторный ANOVA (регион + пол):")
print(anova2)

# Визуализация: boxplot BMI по регионам и полу
plt.figure(figsize=(9,5))
sns.boxplot(data=df, x='region', y='bmi', hue='sex', order=sorted(df['region'].unique()))
plt.title("BMI по регионам и полу")
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.legend(title="Пол")
plt.show()

# === 6. Пост-хок Тьюки для групп (регион+пол) ===
df['group'] = df['region'] + "_" + df['sex']
tukey2 = pairwise_tukeyhsd(endog=df['bmi'], groups=df['group'], alpha=0.05)
print("\nТест Тьюки (регион+пол):")
print(tukey2)

# График доверительных интервалов различий (Тьюки, регион+пол)
fig2 = tukey2.plot_simultaneous(figsize=(10,6))
plt.title("Пост-хок Тьюки: различия BMI между группами (регион+пол)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
