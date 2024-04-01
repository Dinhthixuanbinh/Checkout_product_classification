from tools.process_input import convert_box, rotate_cropped_img
from tools.process_output import calculate_acc, model_pred
from tools.create_database import create_module
from tqdm import tqdm
from PIL import Image
import argparse
import os


def evaluate_data(image_folder,
                  label_folder,
                  embedding_model,
                  id_class_dict,
                  database):
    print('Start evaluate clutter data')
    for sub_folder in os.listdir(image_folder):
        acc_list = []
        sub_path = os.path.join(image_folder, sub_folder)
        for img_name in tqdm(os.listdir(sub_path)):
            label_name = img_name.replace('.jpg', '.txt')
            img_path = os.path.join(sub_path, img_name)
            label_path = os.path.join(label_folder, label_name)

            # Open the label file and read its contents
            with open(label_path, 'r') as file:
                lines = file.readlines()
            # Open the image
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Iterate through each line in the label file
            for iter, line in enumerate(lines):
                # Extract class label and bounding box coordinates
                class_id, x_center, y_center, width, height = map(float,
                                                                  line.split())

                # Convert normalized coordinates to absolute coordinates
                left, top, right, bottom = convert_box(x_center, y_center,
                                                       width, height,
                                                       img_height, img_width)
                # Crop the image
                cropped_img = img.crop((left, top, right, bottom))
                W, H = cropped_img.size

                if W > H:
                    cropped_img = rotate_cropped_img(img_path,
                                                     label_path, iter)

                index_pred = model_pred(embedding_model, cropped_img,
                                        class_id, database, id_class_dict)
                if int(index_pred) == int(class_id):
                    acc_list.append(1)
                else:
                    acc_list.append(0)
        print(f'Accuracy of {sub_folder} clutter {calculate_acc(acc_list)}%')


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate Dataset')
    parser.add_argument("--image_folder", type=str,
                        default=r'val_level_img')
    parser.add_argument("--label_folder", type=str,
                        default=r'labels_val')
    parser.add_argument("--directory_path", type=str,
                        default=r'label_crop_top10')
    parser.add_argument("--id_class_path", type=str,
                        default=r'map_id.csv')
    opt = parser.parse_args()
    return opt


def main(opt):
    embedd_model, database, id_class_dict = create_module(opt.directory_path,
                                                          opt.id_class_path)
    evaluate_data(opt.image_folder, opt.label_folder,
                  embedd_model, id_class_dict, database)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
