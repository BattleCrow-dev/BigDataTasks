import time
import pandas as pd
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Загрузка данных
df = pd.read_csv('recipeData.csv', encoding='latin1')
print("Размер датасета:", df.shape)

# 2. Поиск целевой колонки (интеллектуальный выбор)
target = 'Style'

print("Выбрана целевая колонка:", target)
print("Число классов:", df[target].nunique())

# 3. Предобработка: выберем признаки
num_cols_candidates = ['ABV', 'IBU', 'OG', 'FG', 'BoilGravity', 'BoilTime', 'BatchSize']
num_cols = [c for c in num_cols_candidates if c in df.columns]

text_col = 'BrewMethod'

print("Числовые колонки:", num_cols)
print("Текстовая колонка для TF-IDF:", text_col)

# 4. Упростим задачу: оставим топ-K классов по частоте
K = 10
top_classes = df[target].value_counts().nlargest(K).index.tolist()
df = df[df[target].isin(top_classes)].copy()
df[target] = df[target].astype(str)
print("Оставлено классов:", df[target].nunique(), " (топ {})".format(K))

# 5. Удалим строки с пропусками в целевой и подготовим X, y
df = df.dropna(subset=[target])
X = df.copy()
y = X.pop(target)

# 6. Простая обработка числовых признаков: заполнение и масштабирование
# 7. Текстовые признаки: TF-IDF (ингредиенты)
numeric_transformer = Pipeline(steps=[
    ('imputer', StandardScaler())
])

# TF-IDF для ингредиентов
if text_col is not None:
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
else:
    tfidf = None

# Сформируем список колонок для ColumnTransformer
numeric_cols_present = [c for c in num_cols if c in X.columns]
for c in numeric_cols_present:
    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

# ColumnTransformer: числовые и текстовые
transformers = []
if numeric_cols_present:
    transformers.append(('num', StandardScaler(), numeric_cols_present))
if text_col is not None:
    transformers.append(('tfidf', tfidf, text_col))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)

# 8. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=42)
print("Train/Test:", X_train.shape, X_test.shape)

# 9. Модели: Bagging (RandomForest и Bagging with DecisionTree) и Boosting (CatBoost)
results = {}

# Helper: обучение и оценка
def fit_and_eval(name, pipeline, X_train, y_train, X_test, y_test):
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    t0 = time.time()
    y_pred = pipeline.predict(X_test)
    predict_time = time.time() - t0
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} — время обучения: {train_time:.2f}s, предсказание: {predict_time:.2f}s, F1_macro: {f1:.4f}, Accuracy: {acc:.4f}")
    return {'model': pipeline, 'train_time': train_time, 'predict_time': predict_time, 'f1': f1, 'acc': acc, 'y_pred': y_pred}

# 9.1 RandomForest (бэггинг-подход)
rf = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=1, n_jobs=-1))
])
results['RandomForest'] = fit_and_eval('RandomForest', rf, X_train, y_train, X_test, y_test)

# 9.2 Bagging с деревом (BaggingClassifier)
bag_tree = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=15),
                              n_estimators=200, random_state=1, n_jobs=-1))
])

results['BaggingTree'] = fit_and_eval('Bagging (DecisionTree)', bag_tree, X_train, y_train, X_test, y_test)

# 9.3 Boosting: CatBoost
print("\nОбучаем CatBoostClassifier (будет использовать CPU)")

# Преобразуем признаки
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# CatBoost принимает dense numpy
X_train_cb = X_train_trans.toarray() if hasattr(X_train_trans, "toarray") else X_train_trans
X_test_cb = X_test_trans.toarray() if hasattr(X_test_trans, "toarray") else X_test_trans
cb_model = cb.CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=100, random_state=1)
t0 = time.time()
cb_model.fit(X_train_cb, y_train)
train_time = time.time() - t0
t0 = time.time()
y_pred_cb = cb_model.predict(X_test_cb)
predict_time = time.time() - t0
f1_cb = f1_score(y_test, y_pred_cb, average='macro')
acc_cb = accuracy_score(y_test, y_pred_cb)
print(f"\nCatBoost — время обучения: {train_time:.2f}s, предсказание: {predict_time:.2f}s, F1_macro: {f1_cb:.4f}, Accuracy: {acc_cb:.4f}")
results['CatBoost'] = {'model': cb_model, 'train_time': train_time, 'predict_time': predict_time, 'f1': f1_cb, 'acc': acc_cb, 'y_pred': y_pred_cb}

# 10. Сравнение результатов: таблица
summary = []
for name, res in results.items():
    summary.append({
        'model': name,
        'train_time_s': res['train_time'],
        'predict_time_s': res['predict_time'],
        'f1_macro': res['f1'],
        'accuracy': res['acc']
    })
summary_df = pd.DataFrame(summary).sort_values(by='f1_macro', ascending=False).reset_index(drop=True)

# 11. Подробный отчёт для лучшей модели
best = summary_df.loc[0, 'model']
print("Лучшая модель по F1_macro:", best)
best_res = results[best]
y_pred_best = best_res['y_pred']
print("\nClassification report (best model):")
print(classification_report(y_test, y_pred_best, digits=4))

# 12. Матрица ошибок
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred_best, labels=sorted(y.unique()))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion matrix — {best}')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# 13. Выводы (автоматически сформированные)
print("\nКраткие выводы:")
for idx, row in summary_df.iterrows():
    print(f"- {row['model']}: F1_macro={row['f1_macro']:.4f}, accuracy={row['accuracy']:.4f}, train_time={row['train_time_s']:.1f}s")

print("\nРекомендации:")
print("- Если важна скорость обучения, выбирайте модель с меньшим временем train_time при приемлемом F1.")
print("- Для максимального качества на табличных данных с текстовыми ингредиентами CatBoost/XGBoost обычно дают лучший баланс качества/времени.")
print("- Для дальнейшего улучшения: гиперпараметрический поиск (GridSearch/RandomizedSearch), более тщательная обработка текста (удаление шума, лемматизация), использование признаков взаимодействия и отбора признаков.")
