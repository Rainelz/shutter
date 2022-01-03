import numpy.random as random
import PIL

from spoilers.abstract_filter import AbstractFilter


class Dilate(AbstractFilter):
    """Dilate black blobs in component."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.morphs = [cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE]

    def get_kernel(self):
        # morph = random.choice(self.morphs)
        ksize = random.choice([1, 3])
        # return cv2.getStructuringElement(morph, (ksize, ksize))
        return ksize

    def run(self, image):
        kernel = self.get_kernel()
        # image = cv2.erode(np.array(image),kernel,iterations=2)
        return image.filter(PIL.ImageFilter.MinFilter(kernel))
