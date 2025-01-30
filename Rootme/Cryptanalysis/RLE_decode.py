from PIL import Image, ImageDraw

def decode_rle(expression):
    parts = expression.split('+')
    binary_str = ''
    for part in parts:
        color, count = part.split('x')
        if color == '0':
            binary_str += '0' * int(count)
        elif color == '1':
            binary_str += '1' * int(count)
        else:
            raise ValueError(f"Invalid color code: {color}")
    return binary_str

expressions = [

]

decoded_lines = [decode_rle(expr) for expr in expressions]

max_width = max(len(line) for line in decoded_lines)
height = len(decoded_lines)

normalized_lines = [line.ljust(max_width, '0') for line in decoded_lines]

img = Image.new('1', (max_width, height), color=1)
draw = ImageDraw.Draw(img)

for y, line in enumerate(normalized_lines):
    for x, bit in enumerate(line):
        if bit == '1':
            img.putpixel((x, y), 0)

img.show()

img.save("password_image.png")
print("saved as: 'password_image.png'.")