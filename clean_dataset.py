import os
from PIL import Image

dataset_path = "dataset/train"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Removing:", path)
            os.remove(path)

print("Dataset cleaned!")