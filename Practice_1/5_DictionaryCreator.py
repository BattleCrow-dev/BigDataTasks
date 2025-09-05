A = [1, 2, 3, 4, 2, 1, 3, 4, 5, 6, 5, 4, 3, 2]
B = ['a', 'b', 'c', 'c', 'c', 'b', 'a', 'c', 'a', 'a', 'b', 'c', 'b', 'a']

answer_dictionary = {}

for i in range(len(B)):
    if B[i] in answer_dictionary.keys():
        answer_dictionary[B[i]] += A[i]
    else:
        answer_dictionary[B[i]] = A[i]

print(answer_dictionary)
