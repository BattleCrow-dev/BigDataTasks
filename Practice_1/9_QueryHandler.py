operations = {
    "read": "r",
    "write": "w",
    "execute": "x"
}

n = int(input("Enter the files count: "))
files = {}

for _ in range(n):
    parts = input().split()
    filename = parts[0]
    perms = set(parts[1:]) if len(parts) > 1 else set()
    files[filename] = perms

m = int(input("Enter the queries count: "))

for _ in range(m):
    op, filename = input().split()
    code = operations[op]
    if code in files.get(filename, set()):
        print("OK")
    else:
        print("Access denied")
