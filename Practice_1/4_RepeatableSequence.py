N = int(input("Enter the N: "))
answer_list = []
number, counter = 1, 0

for i in range(N):
    answer_list.append(number)
    counter += 1
    if counter == number:
        counter = 0
        number += 1

print(*answer_list)
