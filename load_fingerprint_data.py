from PIL import Image
import numpy as np

margin = 5
people = 600

width = 96
height = 103


# loads image and saves pixel data in numpy array
def load_image(path):
    im = Image.open(path)
    w, h = im.size
    if w != width or h != height:
        im = im.resize((width, height), Image.ANTIALIAS)
    arr = np.zeros((width - (2 * margin), height - (2 * margin)))
    for j in range(margin, height - margin):
        for i in range(margin, width - margin):
            # Get Pixel
            pixel = im.getpixel((i, j))

            gray = pixel[0]

            arr[i - margin, j - margin] = gray
    return arr


fingers = ["_Left_index_finger.BMP", "_Left_little_finger.BMP", "_Left_middle_finger.BMP",
           "_Left_ring_finger.BMP", "_Left_thumb_finger.BMP", "_Right_index_finger.BMP",
           "_Right_little_finger.BMP", "_Right_middle_finger.BMP",
           "_Right_ring_finger.BMP", "_Right_thumb_finger.BMP"]


# load fingers, returns numpy array of fingers and labels
def load_fingers_data():
    print("start loading data")
    X = []
    y = []
    for i in range(1, people + 1):
        for finger in range(10):
            try:
                x = load_image("SOCOFing/Real/" + str(i) + '__' + 'M' + fingers[finger])
                y.append(0)
            except IOError:
                x = load_image("SOCOFing/Real/" + str(i) + '__' + 'F' + fingers[finger])
                y.append(1)
            X.append(x)
    X = np.array(X)
    y = np.array(y)
    print("done loading data")
    return X, y
