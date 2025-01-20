data = open('ch7.bin', 'rb').read()
out = ''
for d in data:
    out += chr(d - 10)
print(out)