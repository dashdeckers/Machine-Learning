import cv2
import numpy as np

im1 = cv2.imread('im1.png')
im2 = cv2.imread('im2.png')
im3 = cv2.imread('im3.png')
im4 = cv2.imread('im4.png')
im5 = cv2.imread('im5.png')
imlist = [im1, im2, im3, im4, im5]


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(
    	im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation) for im in im_list]
    return cv2.vconcat(im_list_resize)


im_v_resize = vconcat_resize_min(im_list=imlist)
cv2.imwrite('5best.png', im_v_resize)
