number_1 = float(input("Enter the first number: "))
number_2 = float(input("Enter the second number: "))
operation = input("Enter the operation: ")

match operation:
    case '+':
        print(number_1 + number_2)
    case '-':
        print(number_1 - number_2)
    case '/':
        print(number_1 / number_2)
    case '//':
        print(number_1 // number_2)
    case 'abs':
        print(abs(number_1), abs(number_2))
    case 'pow':
        print(pow(number_1, number_2))
    case _:
        print("Unavailable operation")
