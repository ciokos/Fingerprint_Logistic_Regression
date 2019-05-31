import numpy as np

from load_fingerprint_data import load_fingers_data


def save_binary(threshold, x_label, y_label):
    X, y = load_fingers_data()
    print("processing images")
    prepare(X, threshold)
    print("saving")
    np.save(x_label, X)
    np.save(y_label, y)
    print("saved")


A0 = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60,
      62, 63, 96, 112, 120, 124, 126, 127, 129, 131, 135,
      143, 159, 191, 192, 193, 195, 199, 207, 223, 224,
      225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
      251, 252, 253, 254]

A1 = [7, 14, 28, 56, 112, 131, 193, 224]

A2 = [7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135,
      193, 195, 224, 225, 240]

A3 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120,
      124, 131, 135, 143, 193, 195, 199, 224, 225, 227,
      240, 241, 248]

A4 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
      124, 126, 131, 135, 143, 159, 193, 195, 199, 207,
      224, 225, 227, 231, 240, 241, 243, 248, 249, 252]

A5 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
      124, 126, 131, 135, 143, 159, 191, 193, 195, 199,
      207, 224, 225, 227, 231, 239, 240, 241, 243, 248,
      249, 251, 252, 254]

A1pix = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56,
         60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131,
         135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
         224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
         249, 251, 252, 253, 254]

mask = [128, 1, 2, 4, 8, 16, 32, 64]


def prepare(X, threshold):
    for i in range(len(X)):
        thinning(X[i], threshold)
    print("done preparing the data")


def binarize(image, threshold):
    width, height = image.shape
    for j in range(height):
        for i in range(width):
            new_value = 0
            if image[i, j] < threshold:
                new_value = 1

            image[i, j] = new_value
    delete_single_pixels(image)


def delete_single_pixels(image):
    width, height = image.shape
    for j in range(height):
        for i in range(width):
            if i == 0 or i == width - 1 or j == 0 or j == height - 1:
                image[i, j] = 0
                continue
            if image[i - 1, j - 1] == 1:
                continue
            if image[i - 1, j] == 1:
                continue
            if image[i - 1, j + 1] == 1:
                continue
            if image[i, j - 1] == 1:
                continue
            if image[i, j + 1] == 1:
                continue
            if image[i + 1, j - 1] == 1:
                continue
            if image[i + 1, j] == 1:
                continue
            if image[i + 1, j + 1] == 1:
                continue
            image[i, j] = 0


def thinning(image, threshold):
    binarize(image, threshold)
    changed = True
    while changed:
        changed = False
        phase(A0, 2, 1, image)
        if phase(A1, 0, 2, image):
            changed = True
        if phase(A2, 0, 2, image):
            changed = True
        if phase(A3, 0, 2, image):
            changed = True
        if phase(A4, 0, 2, image):
            changed = True
        if phase(A5, 0, 2, image):
            changed = True
        phase6(image)
    phase(A1pix, 0, 2, image)


def phase(A, newValue, focus, bitmap):
    changed = False
    width, height = bitmap.shape
    for y in range(height):
        for x in range(width):
            if bitmap[x, y] == focus:
                weight = useMask(y, x, bitmap)
                if weight in A:
                    bitmap[x, y] = newValue
                    changed = True

    return changed


def phase6(bitmap):
    w, h = bitmap.shape
    for y in range(h):
        for x in range(w):
            if bitmap[x, y] > 0:
                bitmap[x, y] = 1


def useMask(y, x, bitmap):
    neighbours = []
    w, h = bitmap.shape
    if x - 1 >= 0 and y - 1 >= 0:
        neighbours.append(bitmap[x - 1, y - 1])
    else:
        neighbours.append(0)
    if y - 1 >= 0:
        neighbours.append(bitmap[x, y - 1])
    else:
        neighbours.append(0)
    if x + 1 < w and y - 1 >= 0:
        neighbours.append(bitmap[x + 1, y - 1])
    else:
        neighbours.append(0)
    if x + 1 < w:
        neighbours.append(bitmap[x + 1, y])
    else:
        neighbours.append(0)
    if y + 1 < h and x + 1 < w:
        neighbours.append(bitmap[x + 1, y + 1])
    else:
        neighbours.append(0)
    if y + 1 < h:
        neighbours.append(bitmap[x, y + 1])
    else:
        neighbours.append(0)
    if y + 1 < h and x - 1 >= 0:
        neighbours.append(bitmap[x - 1, y + 1])
    else:
        neighbours.append(0)
    if x - 1 >= 0:
        neighbours.append(bitmap[x - 1, y])
    else:
        neighbours.append(0)
    return weight(neighbours)


def weight(list):
    summ = 0
    for i in range(len(list)):
        if list[i] > 0:
            summ += mask[i]
    return summ
