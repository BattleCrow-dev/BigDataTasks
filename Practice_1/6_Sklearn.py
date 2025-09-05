from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame

print("--> Use info()")
print(df.info())
print("------\n")

print("--> Use isna().sum()")
print(df.isna().sum())
print("------\n")

print("--> Average age")
print(df.loc[(df.HouseAge > 50) & (df.Population > 2500)])
print("------\n")

print("--> Maximum and minimum cost")
print('max = ', max(df.MedHouseVal), '\nmin = ', min(df.MedHouseVal))
print("------\n")

print("--> Use apply()")
print(df.apply(lambda col: col.mean()))
print("------\n")