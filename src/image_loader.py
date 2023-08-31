from typing import List, Tuple
from PIL import Image, ImageOps
import numpy as np
import glob
import matplotlib.pyplot as plt
import image_processor


#[([pixels], number), (), ()]


class ImageLoader(object):
    def __init__(self, image_size: Tuple[int, int], grid_size: Tuple[int, int], grid_directory: str):
        self.IMAGE_SIZE = image_size
        self.GRID_SIZE = grid_size
        self.grid_directory = grid_directory
        self.image_processor = image_processor.ImageProcessor(self.IMAGE_SIZE)
        self.images = []
        self.frame = self.create_frame(3)

        

    def loadGridsAndSaveDigits(self):
        ims = glob.glob(f"{self.grid_directory}/*")
        empty = np.empty(self.IMAGE_SIZE)

        for image in ims:
            with Image.open(image) as im:
                im = im.convert("L")
                im = ImageOps.invert(im)
                im = im.resize(self.GRID_SIZE)
                filename = self.getImageName(image)

                pixels = list(im.getdata())
                pixels = [pixel/255 for pixel in pixels]
                pixels = np.reshape(pixels, self.GRID_SIZE)

                for n in range(10):
                    for m in range(10):
                        im = empty

                        for i in range(self.IMAGE_SIZE[0]):
                            for j in range(self.IMAGE_SIZE[1]):
                                x = self.IMAGE_SIZE[0]*n + i
                                y = self.IMAGE_SIZE[1]*m + j

                                im[i, j] = pixels[x, y]

                        im = im * self.frame
                        im = self.image_processor.process(im)
                        self.images.append((im, n))
                        plt.imsave(f"images/digits/im({filename})_{n},{m}.jpg", im, cmap="gray")

    def getImageName(self, file_location: str):
        return file_location.split('/')[-1].split(".")[0]


    def create_frame(self, border_width: int):
        frame = np.zeros(self.IMAGE_SIZE)

        for i in range(self.IMAGE_SIZE[0]):
            for j in range(self.IMAGE_SIZE[1]):
                if (i > border_width-1 and i < 28-border_width and j > border_width-1 and j < 28-border_width):
                    frame[i][j] = 1.0

        return frame

    def print_frame(self, frame): 
        for i in range(self.IMAGE_SIZE[0]):
            for j in range(self.IMAGE_SIZE[1]):
                print(frame[i][j], end="  ")
            print()
            print()

        
    





