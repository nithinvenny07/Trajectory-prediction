import cv2
import numpy as np
import glob


def generate():
    img_array = []
    for filename in glob.glob('output/*.jpg'):
    	img = cv2.imread(filename)
    	height, width, layers = img.shape
    	size = (width, height)
    	img_array.append(img)
    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()