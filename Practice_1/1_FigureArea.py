import math

chosen_figure = input("Enter the type of figure (triangle, recrangle, circle): ")
area = 0

if chosen_figure == 'triangle':
    side = float(input("Enter the side: "))
    height = float(input("Enter the height: "))
    area = 0.5 * side * height
elif chosen_figure == 'rectangle':
    side_1 = float(input("Enter the sides: "))
    side_2 = float(input("Enter the sides: "))
    area = side_1 * side_2
elif chosen_figure == 'circle':
    radius = float(input("Enter the radius: "))
    area = math.pi * radius**2
else:
    area = 'Unavailable figure'

print({chosen_figure : area})