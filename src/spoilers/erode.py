import numpy.random as random
import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class Erode(AbstractFilter):
    """Erode black blobs in component."""

    DEFAULT_K = 3

    def __init__(self, k=DEFAULT_K, **kwargs):
        super(Erode, self).__init__(**kwargs)
        self.k = k
        # self.morphs = [cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE]

    def get_kernel(self):
        # morph = random.choice(self.morphs)
        ksize = random.choice([1, 3])
        # return cv2.getStructuringElement(morph, (ksize, ksize))
        return ksize

    def run(self, image):
        # kernel = self.get_kernel()
        k = roll_value(self.k)
        data = {"type": self.type(), "k": k}
        self.annotate(image, data)
        # image = cv2.erode(np.array(image),kernel,iterations=2)
        return image.filter(PIL.ImageFilter.MaxFilter(self.k))
