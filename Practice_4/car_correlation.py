import matplotlib.pyplot as plt
import numpy as np

# Данные
street = np.array([80, 98, 75, 91, 78])
garage = np.array([100, 82, 105, 89, 102])

# Корреляция Пирсона
corr = np.corrcoef(street, garage)[0, 1]
print(f"Коэффициент корреляции Пирсона: {corr:.2f}")

# Диаграмма рассеяния
plt.scatter(street, garage, color='blue', label='Дни недели')
plt.title("Диаграмма рассеяния: Улица vs Гараж")
plt.xlabel("Количество машин на улице")
plt.ylabel("Количество машин в гараже")
plt.grid(True)
plt.legend()
plt.show()
