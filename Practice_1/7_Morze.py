morze = {'a': '.-', 'b': '-…', 'c': '-.-.', 'd': '-..',
'e': '.', 'f': '..-.', 'g': '--.', 'h': '....',
'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
'm': '--', 'n': '-.', 'o': '---', 'p': '.--.',
'q': '--.-', 'r': '.-.', 's': '…', 't': '-',
'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-',
'y': '-.--', 'z': '--..'}

input_string = input("Enter the text to translate: ")
result_string = ""

for letter in input_string:
    if letter == ' ':
        result_string += '\n'
    elif letter in morze.keys():
        result_string += morze[letter.lower()] + ' '

print(result_string)