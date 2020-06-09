import cv2
import numpy as np
from pickle import dump

def formatToTxt(data, imagePath, filePath):
  images = {}
  annots = []

  classNames = ['tops', 'trousers', 'outerwear', 'dresses', 'skirts']

  with open(filePath, 'wb') as f:
    total = len(data['annotations'])
    img2Scales = {}

    for i, annotation in enumerate(data['annotations']):
        filePath = imagePath + str(annotation['image_id']) + '.jpg'
        imageId = annotation['image_id']
        x1 = annotation['bbox'][0]
        y1 = annotation['bbox'][1]
        x2 = annotation['bbox'][2] + x1
        y2 = annotation['bbox'][3] + y1
        objClass = classNames[annotation['category_id']-1]        
        if not imageId in images:
            img = cv2.imread(filePath, cv2.IMREAD_COLOR)
            minDim = 0 if img.shape[0] < img.shape[1] else 1
            newDim = [None, None]
            newDim[minDim] = 299
            # new dim's [minDim] should be 299 * original[minDim]
            newDim[1-minDim] = int(float(img.shape[1-minDim]) / float(img.shape[minDim]) * 299)
            img2Scales[imageId] = [float(newDim[0]) / float(img.shape[0]), float(newDim[1]) / float(img.shape[1])]
            # resize image
            img = cv2.resize(img, tuple(newDim))
            images[imageId] = img
        # rescale bbox
        sx, sy = img2Scales[imageId]
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy
        annots.append([imageId, x1, y1, x2, y2, objClass])
        print('Completed : %03d / %03d' % (i, total))
    dump({ 'annots' : annots, 'images' : images}, f)
