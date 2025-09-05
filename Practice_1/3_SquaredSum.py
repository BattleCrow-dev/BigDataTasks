current_sum = 0
current_squared_sum = 0

while True:
    number = float(input("Enter the number: "))
    current_sum += number
    current_squared_sum += number**2
    if current_sum == 0: break

print("Squared_sum = ", current_squared_sum)