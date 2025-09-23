import pandas as pd

# 9. Загрузка данных
df = pd.read_csv("ECDCCases.csv")

# 10. Проверка пропусков
print("Количество пропусков в процентах по каждому признаку:")
missing_percent = df.isnull().mean() * 100
print(missing_percent)

# Определяем два признака с наибольшим количеством пропусков
cols_to_drop = missing_percent.sort_values(ascending=False).head(2).index
print(f"\nУдаляем признаки с наибольшим количеством пропусков: {list(cols_to_drop)}")
df = df.drop(columns=cols_to_drop)

# Обработка оставшихся пропусков
for col in df.columns:
    if df[col].dtype == "object":  # категориальный признак
        df[col] = df[col].fillna("other")
    else:  # числовой признак
        df[col] = df[col].fillna(df[col].median())

# Проверяем, что пропусков больше нет
print("\nПроверка после обработки:")
print(df.isnull().sum().sum(), "пропусков осталось")

# 11. Статистика по данным
print("\nОписательная статистика:")
print(df.describe(include="all"))

# Выводы о выбросах:
# Обычно выбросы видны по сильно отличающимся max от 75% перцентиля.
# Например, если deaths или cases имеют очень большие значения.

# Для каких стран количество смертей в день превысило 3000
if "deaths" in df.columns and "countriesAndTerritories" in df.columns:
    high_deaths = df[df["deaths"] > 3000]
    result = high_deaths.groupby("countriesAndTerritories").size()
    print("\nСтраны и количество дней, когда смертей > 3000:")
    print(result)

# 12. Поиск и удаление дубликатов
duplicates = df.duplicated().sum()
print(f"\nКоличество дубликатов в данных: {duplicates}")

df = df.drop_duplicates()
print(f"После удаления дубликатов: {df.duplicated().sum()} дубликатов")
