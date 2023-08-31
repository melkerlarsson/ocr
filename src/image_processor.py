class ImageProcessor(object):


    def __init__(self, size):
        self.size_x = size[0]
        self.size_y = size[1]
        pass
      
    
    def process(self, pixels):
        for i in range(self.size_x ):
            for j in range(self.size_y ):
                p = pixels[i][j]
                if p < 0.40:
                    p = 0
                elif p > 0.50:
                    p = 1
                else:
                    p = 0.5

                pixels[i][j] = p
        
        return pixels
                