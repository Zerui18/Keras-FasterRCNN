import numpy as np
from pickle import load

class_names = ['tops', 'trousers', 'outerwear', 'dresses', 'skirts']

def get_data(pickle_path):
    all_imgs = {}

    class_mapping = { c : i for i, c in enumerate(class_names) }

    data = load(open(pickle_path, 'rb'))

    classes_count = {}

    for annot in data['annots']:

        (image_id, x1, y1, x2, y2, class_name) = annot

        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if image_id not in all_imgs:
            image = {}
            image_data = data['images'][image_id]
            (rows, cols) = image_data.shape[:2]
            image['data'] = image_data
            image['width'] = cols
            image['height'] = rows
            image['bboxes'] = []
            image['imageset'] = 'train'
            all_imgs[image_id] = image

        all_imgs[image_id]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    return all_data, classes_count, class_mapping
