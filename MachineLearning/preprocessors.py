from keras.preprocessing.image import img_to_array
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)
