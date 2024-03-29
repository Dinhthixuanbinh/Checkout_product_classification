import os
import shutil
import random
import json
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from PIL import Image

import argparse

DATA_ROOT = '../datasets/retail_product_checkout_dataset'
WORK_ROOT = '../datasets/retail_product_checkout_dataset_labels'

# Check if DATA_ROOT exists and list its contents
if os.path.exists(DATA_ROOT):
    print(f"Contents of {DATA_ROOT}:")
    os.system(f'ls -l {DATA_ROOT}')
else:
    print(f"{DATA_ROOT} does not exist.")

# Create WORK_ROOT if it doesn't exist
if not os.path.exists(WORK_ROOT):
    os.makedirs(WORK_ROOT)

class RetailProductDatasetConverter:
    def __init__(self, data_root, work_root):
        self.data_root = data_root
        self.work_root = work_root
        self.nc = 0
        self.names = []
    def load_json(self, jfile):
        with open(jfile, 'rb') as f:
            return json.load(f)

    def get_unique_sku_class_list(self, val_data):
        val_rawcat_df = pd.DataFrame(val_data['__raw_Chinese_name_df'])
        unique_sku_class = val_rawcat_df['sku_class'].unique()
        self.nc = len(unique_sku_class)
        self.names = unique_sku_class.tolist()
        return self.nc, self.names

    def convert_annotation(self, phase):
        in_dir = self.data_root
        out_dir = os.path.join(self.work_root, 'out')
        in_images_path = f'{in_dir}/{phase}2019'
        out_images_path = f'{out_dir}/images/{phase}'
        out_labels_path = f'{out_dir}/labels/{phase}'

        os.makedirs(out_images_path, exist_ok=True)
        os.makedirs(out_labels_path, exist_ok=True)

        # Load json description
        with open(os.path.join(in_dir, f'instances_{phase}2019.json'), 'rb') as f:
            data = json.load(f)

        imgs_df = pd.DataFrame(data['images'])
        anns_df = pd.DataFrame(data['annotations'])
        class_name = pd.DataFrame(data['__raw_Chinese_name_df'])

        # Filter category id
        categoryid_list = class_name['category_id']
        anns_df = anns_df[anns_df['category_id'].isin(categoryid_list)]

        # Filter image id
        img_ids = anns_df['image_id']
        imgs_df = imgs_df[imgs_df['id'].isin(img_ids)]
        # Load validation data
        val_data = self.load_json(os.path.join(self.data_root, 'instances_val2019.json'))
        self.nc, self.names = self.get_unique_sku_class_list(val_data)
        
        # Convert to YOLOv5 format
        for _, item in imgs_df.iterrows():
            imgw, imgh = item.width, item.height
            dw, dh = 1.0 / imgw, 1.0 / imgh
            img_src_path = os.path.join(in_images_path, item.file_name)
            if not os.path.exists(img_src_path):
                continue
            img_dst_path = os.path.join(out_images_path, item.file_name)
            lab_dst_path = os.path.join(out_labels_path, item.file_name.replace('.jpg', '.txt'))

            # Annotation in this image
            anns = anns_df[anns_df['image_id'] == item.id]
            labs = []
            for _, ann in anns.iterrows():
                ann_class_name = class_name[class_name['category_id'] == ann.category_id]
                cls_id = self.get_unique_sku_class_list(ann_class_name)[0]
                cx, cy = dw * ann.point_xy[0], dh * ann.point_xy[1]
                sw, sh = dw * ann.bbox[2], dh * ann.bbox[3]
                labs.append(f'{cls_id} {cx:.6f} {cy:.6f} {sw:.6f} {sh:.6f}')

            # Save labels and copy image
            with open(lab_dst_path, 'w') as fw:
                fw.write('\n'.join(labs))
            shutil.copyfile(img_src_path, img_dst_path)
    def parse_args():
        parser = argparse.ArgumentParser(description='Convert the retail product checkout dataset to YOLOv8 format.')
        parser.add_argument('--data-root', type=str, default=DATA_ROOT, help='Path to the dataset root directory')
        parser.add_argument('--work-root', type=str, default=WORK_ROOT, help='Path to the working directory')
        return parser.parse_args()
    def main():
        # Parse command-line arguments
        args = parse_args()
        DATA_ROOT = args.data_root
        WORK_ROOT = args.work_root

        # Load validation data
        val_data = converter.load_json(os.path.join(DATA_ROOT, 'instances_val2019.json'))
        
        # Initialize converter
        converter = RetailProductDatasetConverter(DATA_ROOT, WORK_ROOT)

        # Convert annotations
        unique_sku_class_list = converter.get_unique_sku_class_list(val_data)
        converter.convert_annotation('train', unique_sku_class_list)
        converter.convert_annotation('val', unique_sku_class_list)
        converter.convert_annotation('test', unique_sku_class_list)
