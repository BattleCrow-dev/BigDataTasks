n = int(input("Enter the line count: "))
database = {}

for i in range(n):
    name = input("Enter the name: ")

    if name not in database:
        database[name] = 1
        print("OK")
    else:
        print(name + str(database[name]))
        database[name] += 1