import pandas as pd
from scipy.stats import shapiro, bartlett, ttest_ind

# Загрузка данных
bmi_df = pd.read_csv("bmi.csv")

# Выборки по регионам
northwest_bmi = bmi_df[bmi_df["region"] == "northwest"]["bmi"]
southwest_bmi = bmi_df[bmi_df["region"] == "southwest"]["bmi"]

# Проверка нормальности (Шапиро-Уилка)
shapiro_nw = shapiro(northwest_bmi)
shapiro_sw = shapiro(southwest_bmi)
print("Шапиро-Уилка для northwest:", shapiro_nw)
print("Шапиро-Уилка для southwest:", shapiro_sw)

# Проверка гомогенности дисперсий (Бартлетт)
bartlett_test = bartlett(northwest_bmi, southwest_bmi)
print("Критерий Бартлетта:", bartlett_test)

# Сравнение средних значений (t-критерий Стьюдента)
ttest_result = ttest_ind(northwest_bmi, southwest_bmi, equal_var=True)
print("t-критерий Стьюдента:", ttest_result)
