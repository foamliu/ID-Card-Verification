import math

import cv2 as cv
import numpy as np
from PIL import Image

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import image_h, image_w
from mtcnn.detector import detect_faces


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_face_all_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            i = select_central_face(img.size, bounding_boxes)
            return True, [bounding_boxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None, None


def select_central_face(im_size, bounding_boxes):
    width, height = im_size
    nearest_index = -1
    nearest_distance = 100000
    for i, b in enumerate(bounding_boxes):
        x_box_center = (b[0] + b[2]) / 2
        y_box_center = (b[0] + b[2]) / 2
        x_img = width / 2
        y_img = height / 2
        distance = math.sqrt((x_box_center - x_img) ** 2 + (y_box_center - y_img) ** 2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    return nearest_index
