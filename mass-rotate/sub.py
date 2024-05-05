input_data = "informaticsolympiad"
l = []
for i in range(len(input_data)):
    for j in range(i+1, len(input_data)+1):
        l.append(input_data[i:j])

for i in sorted(list(set(l))):
    print(i)