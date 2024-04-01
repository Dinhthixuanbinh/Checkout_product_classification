from tools.rotate_img import yoloRotatebbox
import cv2
from PIL import Image


def rotate_cropped_img(raw_image, label_path, iter, state=True):
    rotate_obj = yoloRotatebbox(filename=raw_image,
                                angle=-45,
                                labelname=label_path)
    rotated_image = rotate_obj.rotate_image()
    rotated_bbox = rotate_obj.rotateYolobbox(state)
    left, top, right, bottom = map(int, rotated_bbox[iter][1:])
    cropped_img = rotated_image[top:bottom, left:right]
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_img)


def convert_box(x_center, y_center,
                width, height,
                img_H, img_W):
    # Convert normalized coordinates to absolute coordinates
    left = int((x_center - width / 2) * img_W)
    top = int((y_center - height / 2) * img_H)
    right = int((x_center + width / 2) * img_W)
    bottom = int((y_center + height / 2) * img_H)
    return left, top, right, bottom
