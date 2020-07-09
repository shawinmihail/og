from PIL import Image
import numpy as np

n = 1025
d = 20.0
zero_h = 5
map_array = np.empty((n, n), np.float)

def surf(x, y):
    z = 5e-2 * (x*x - y*y) + zero_h
    return z

for i in range(n):
    for j in range(n):
        x = (i - n/2) / float(n) * d
        y = (j - n/2) / float(n) * d
        map_array[i, j] = surf(x, y) / d * 255

print(map_array)
map_img = Image.fromarray(np.uint8(map_array))
map_img.save("image.png", "PNG")