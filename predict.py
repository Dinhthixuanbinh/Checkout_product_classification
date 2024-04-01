from tools.process_input import convert_box, rotate_cropped_img
from tools.process_output import model_pred, get_name_id, plot_bbox, format_lst
from bbox_id_predict import get_bbox
from tools.create_database import create_module
from PIL import Image
import argparse


def inferecne_img(img_path,
                  embedding_model,
                  id_class_dict,
                  database,
                  dataframe,
                  dectector):
    predictions = []
    product_list = []

    img = Image.open(img_path)
    img_width, img_height = img.size
    bbox = get_bbox(img_path, weight_path=dectector, state=False)

    for iter, line in enumerate(bbox):
        class_id, x_center, y_center, width, height = map(float, line.split())
        left, top, right, bottom = convert_box(x_center, y_center,
                                               width, height,
                                               img_height, img_width)
        cropped_img = img.crop((left, top, right, bottom))
        W, H = cropped_img.size
        if W > H:
            cropped_img = rotate_cropped_img(img_path,
                                             bbox, iter,
                                             state=False)
        name_id = get_name_id(dataframe=dataframe,
                              number_id=int(class_id),
                              state=False)
        index_pred = model_pred(embedding_model, cropped_img,
                                name_id, database,
                                id_class_dict, state=True)
        predictions.append((f'{name_id}_{index_pred}',
                            (left, top, right, bottom)))
        product_list.append(f'{name_id}_{index_pred}')

    return product_list, predictions


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict Image')
    parser.add_argument("--directory_path", type=str,
                        default=r'label_crop_top10')
    parser.add_argument("--img_path", type=str,
                        default=r'test_level_img\easy\20180824-13-35-55-2.jpg')
    parser.add_argument("--id_class_path", type=str,
                        default=r'map_id.csv')
    parser.add_argument("--detector", type=str,
                        default=r'best.pt')
    parser.add_argument("--dataframe", type=str,
                        default=r'class_names_with_index.csv')

    opt = parser.parse_args()
    return opt


def main(opt):
    all_module = create_module(opt.directory_path, opt.id_class_path,
                               opt.detector, opt.dataframe)
    proudcts, predictions = inferecne_img(img_path=opt.img_path,
                                          embedding_model=all_module[1],
                                          id_class_dict=all_module[3],
                                          database=all_module[2],
                                          dataframe=all_module[-1],
                                          dectector=all_module[0])
    format_lst(proudcts)
    plot_bbox(opt.img_path, predictions)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
