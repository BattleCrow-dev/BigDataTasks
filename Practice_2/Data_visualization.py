import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# -----------------------------
# 1) Загрузка данных и описание
# -----------------------------

df = pd.read_csv('recipeData.csv', encoding='latin1')

print("Форма датафрейма:", df.shape)
print("Колонки:", df.columns.tolist())

# Описание: насчитывается ~73861 записей и 23 признака.
# Признаки включают: BeerID, Name, Style, Size.L.,
# OG (Original Gravity), FG (Final Gravity), ABV, IBU, Color,
# BoilSize, BoilTime, Efficiency, MashThickness, PrimaryTemp и др.

# -----------------------------
# 2) .info(), .head(), пропуски, предобработка
# -----------------------------

print("\n--- Инфо о данных ---")
df.info()

print("\n--- Первые строки ---")
print(df.head())

# Проверка на пропуски
missing = df.isnull().sum().sort_values(ascending=False)
print("\n--- Пропуски по столбцам (топ 10) ---")
print(missing.head(10))

# Обработка пропусков
# Решение: удалить строки, где отсутствуют критичные числовые признаки
# (например ABV, IBU, OG, FG)
# Или заполнить средними / медианой, если пропусков немного

# Считаем, что ABV обязателен
df_num = df.select_dtypes(include=[np.number])
critical_cols = ['ABV', 'OG', 'FG', 'IBU', 'Color']
for col in critical_cols:
    if col in df_num.columns:
        n_miss = df[col].isnull().sum()
        print(f"Пропусков в {col}: {n_miss}")
        # Если небольшое число пропусков, можно удалить
        # Если большое, можно заполнить средним
        # Здесь пример смешанного подхода
        if n_miss > 0:
            # если менее 5% строк, удаляем
            if n_miss / len(df) < 0.05:
                df = df.dropna(subset=[col])
                print(f"Удалены строки с пропуском в {col}")
            else:
                # заполнение средним
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"Заполнены пропуски в {col} средним {mean_val:.3f}")


# Для визуализации и t-SNE/UMAP нужны числовые признаки — выделим их
num_features = []
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        num_features.append(col)

# Исключим идентификаторы и др. неинформативные
irrelevant = ['BeerID', 'Name', 'StyleID']
num_features = [c for c in num_features if c not in irrelevant]

print("\n--- Числовые признаки, выбранные для анализа ---")
print(num_features)

# Нормализация
scaler = StandardScaler()
df_num_scaled = pd.DataFrame(scaler.fit_transform(df[num_features]), columns=num_features)

# -----------------------------
# 3) Столбчатая диаграмма Plotly (.bar)
# -----------------------------
# Пример: среднее значение ABV по видам пива (Style)
# Для этого группируем по Style, берём среднее ABV

# Но так как видов сотни — выбрать топ-20 по числу элементов
style_counts = df['Style'].value_counts()
top_styles = style_counts.nlargest(20).index.tolist()
bar_df = df[df['Style'].isin(top_styles)].groupby('Style')['ABV'].mean().reset_index()

x_vals = bar_df['Style']
y_vals = bar_df['ABV']

bar = go.Bar(
    x=x_vals,
    y=y_vals,
    marker=dict(
        color=y_vals,
        coloraxis="coloraxis",
        line=dict(color='black', width=2)
    )
)

layout_bar = go.Layout(
    title=dict(text="Среднее ABV в 20 наиболее популярных видах пива", x=0.5, xanchor='center', font=dict(size=20)),
    xaxis=dict(
        title=dict(text='Вид пива', font=dict(size=16)),
        tickangle=315,
        tickfont=dict(size=14),
        showgrid=True, gridwidth=2, gridcolor='ivory'
    ),
    yaxis=dict(
        title=dict(text='Среднее ABV (%)', font=dict(size=16)),
        tickfont=dict(size=14),
        showgrid=True, gridwidth=2, gridcolor='ivory'
    ),
    coloraxis=dict(colorscale='Viridis'),
    width=1200,
    height=700,
    margin=dict(l=20, r=20, t=80, b=120)
)

fig_bar = go.Figure(data=[bar], layout=layout_bar)
fig_bar.show()

# -----------------------------
# 4) Круговая диаграмма go.Pie
# -----------------------------

# Используем распределение видов среди пива (топ 20 как выше) + «Other»
counts_top = style_counts[top_styles]
other_count = style_counts.sum() - counts_top.sum()

labels = list(top_styles) + ['Other']
values = list(counts_top.values) + [other_count]

pie = go.Pie(
    labels=labels,
    values=values,
    marker=dict(line=dict(color='black', width=2))
)

layout_pie = go.Layout(
    title=dict(text="Распределение видов пива (топ-20 + остальные)", x=0.5, font=dict(size=20)),
    width=900,
    height=600,
    margin=dict(l=20, r=20, t=80, b=20)
)

fig_pie = go.Figure(data=[pie], layout=layout_pie)
fig_pie.show()

# -----------------------------
# 5) Линейные графики: зависимость параметров
# -----------------------------

print("\n--- 5) Линейные графики ---")
print("Колонки в датасете:", df.columns.tolist())

# Фильтрация "разумных" значений
df_filtered = df[
    (df['ABV'] <= 15) &
    (df['IBU'] <= 120) &
    (df['OG'].between(1.0, 1.2))
].head(300)  # берём первые 300 строк после фильтрации

x = df_filtered['ABV']

# Задаём параметры для отображения
y_params = {
    'IBU (горечь)': 'IBU',
    'OG (начальная плотность)': 'OG',
    'FG (конечная плотность)': 'FG'
}

plot_params = dict(
    marker='o',
    markersize=6,
    markerfacecolor='white',
    markeredgecolor='black',
    markeredgewidth=2,
    linewidth=2,
    color='crimson'
)

plt.figure(figsize=(12, 6))

for label, col in y_params.items():
    if col in df_filtered.columns:
        y = df_filtered[col]
        plt.plot(x, y, label=label, **plot_params)

plt.xlabel("ABV (содержание алкоголя, %)", fontsize=14)
plt.ylabel("Значения показателей", fontsize=14)
plt.title("Зависимость характеристик пива от содержания алкоголя (ABV)", fontsize=16)

plt.grid(True, linewidth=2, color='mistyrose')
plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

plt.show()

# -----------------------------
# 6) t-SNE визуализация для разных perplexity
# -----------------------------

print("\n--- 6) t-SNE визуализация ---")

# Убираем NaN
df_num_clean = df_num_scaled.dropna()

# Делаем подвыборку, чтобы t-SNE успел посчитать
sample_size = 5000
if len(df_num_clean) > sample_size:
    df_num_clean = df_num_clean.sample(n=sample_size, random_state=42)

perplexities = [5, 30, 50]
tsne_results = {}
total_tsne_time = 0.0

for p in perplexities:
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=p, init='pca',
                learning_rate='auto', random_state=42)
    Z = tsne.fit_transform(df_num_clean)
    t1 = time.time()
    elapsed = t1 - t0
    tsne_results[p] = {'embedding': Z, 'time': elapsed}
    total_tsne_time += elapsed
    print(f"t-SNE (perplexity={p}) — время {elapsed:.2f} сек")

# Построим scatter для разных perplexity
plt.figure(figsize=(18, 5))
for i, p in enumerate(perplexities, start=1):
    Z = tsne_results[p]['embedding']
    ax = plt.subplot(1, len(perplexities), i)
    scatter = ax.scatter(Z[:, 0], Z[:, 1],
                         c=df_num_clean['ABV'],
                         cmap='viridis', s=8)
    ax.set_title(f"perplexity={p}, time={tsne_results[p]['time']:.2f}s")
    ax.axis('off')
plt.suptitle("t-SNE по рецептам пива (цвет = ABV)", fontsize=16)
plt.show()


# -----------------------------
# 7) UMAP визуализация + сравнение скорости
# -----------------------------

print("\n--- 7) UMAP визуализация ---")

n_neighbors_list = [5, 15, 50]
min_dist_list = [0.0, 0.1, 0.5]
umap_results = {}
total_umap_time = 0.0

for n in n_neighbors_list:
    for md in min_dist_list:
        t0 = time.time()
        um = UMAP(n_neighbors=n, min_dist=md,
                  n_components=2, random_state=42)
        U = um.fit_transform(df_num_clean)
        t1 = time.time()
        elapsed = t1 - t0
        umap_results[(n, md)] = {'embedding': U, 'time': elapsed}
        total_umap_time += elapsed
        print(f"UMAP (n_neighbors={n}, min_dist={md}) — время {elapsed:.2f} сек")

# Визуализация
rows = len(n_neighbors_list)
cols = len(min_dist_list)
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
for i, n in enumerate(n_neighbors_list):
    for j, md in enumerate(min_dist_list):
        U = umap_results[(n, md)]['embedding']
        ax = axes[i][j]
        sc = ax.scatter(U[:,0], U[:,1],
                        c=df_num_clean['ABV'], cmap='viridis', s=8)
        ax.set_title(f"n={n}, min_dist={md}\n{umap_results[(n,md)]['time']:.2f}s")
        ax.axis('off')
plt.suptitle("UMAP по рецептам пива (цвет = ABV)", fontsize=16)
plt.show()

print(f"Суммарное время t-SNE: {total_tsne_time:.2f} сек")
print(f"Суммарное время UMAP: {total_umap_time:.2f} сек")