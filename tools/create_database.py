from tools.get_embedding import Img2Vec
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import faiss
import pandas as pd
import os


# Define the path to the main folder
big_folder_path = r'label_crop_top10'
embedding_model = Img2Vec()


def add_embeddings_to_index(embeddings, index):
    faiss.normalize_L2(embeddings)
    index.add(embeddings)


def create_new_database(big_folder_path, embedding_model):
    my_list = []
    # Loop through each subfolder directly inside big_folder
    for subdir1 in tqdm(os.listdir(big_folder_path)):
        index = faiss.IndexFlatIP(1408)
        subdir1_path = os.path.join(big_folder_path, subdir1)
        # Check if the item in big_folder_path is indeed a directory
        if os.path.isdir(subdir1_path):
            # Loop through each sub-subfolder directly inside subdir1
            for subdir2 in os.listdir(subdir1_path):
                subdir2_path = os.path.join(subdir1_path, subdir2)
                if os.path.isdir(subdir2_path):
                    # Loop through each image file in the sub-subfolder
                    for filename in os.listdir(subdir2_path):
                        if filename.endswith(('.jpg', '.jpeg',
                                              '.png', '.gif')):
                            image_path = os.path.join(subdir2_path, filename)
                            img = Image.open(image_path).convert('RGB')
                            embedding_vect = embedding_model.get_vec(img)
                            add_embeddings_to_index(embedding_vect, index)
                            my_list.append(subdir2)

        faiss.write_index(index, f'{subdir1}.index')
        with open(f'{subdir1}.txt', 'w') as f:
            for item in my_list:
                f.write("%s\n" % item)
        my_list = []


def load_database(directory, dimension, model):
    dict_label = {}
    print('Load Database Image')
    for item in tqdm(os.listdir(directory)):
        directorys = os.path.join(directory, item)
        label_list = []
        index = faiss.IndexFlatIP(dimension)

        for sub_folder in os.listdir(directorys):
            sub_folders = os.path.join(directorys, sub_folder)

            for file in os.listdir(sub_folders):
                image_path = os.path.join(sub_folders, file)
                image_path = Image.open(image_path).convert('RGB')
                embedding = model.get_vec(image_path)
                faiss.normalize_L2(embedding)

                index.add(embedding)
                label_list.append(sub_folder)
        dict_label[item] = [index, label_list]
    print("Done Load Database")
    return dict_label


def map_id_class(dict_path):
    df = pd.read_csv(dict_path)
    df = df.drop(columns=['Unnamed: 0'])
    id_supercategory_dict = df.set_index('id')['supercategory'].to_dict()
    return id_supercategory_dict


def get_embedding_size(model,
                       test_path=r"tools\sample.jpg"):
    test_img = Image.open(test_path).convert('RGB')
    test_img = model.get_vec(test_img)
    return test_img.shape[1]


def create_module(directory_path,
                  id_class_path,
                  detector,
                  dataframe,
                  state=True):
    classifier = Img2Vec()
    dimension = get_embedding_size(classifier)
    database = load_database(directory_path,
                             dimension=dimension,
                             model=classifier)
    class_dict = map_id_class(id_class_path)
    df = pd.read_csv(dataframe)
    if state:
        detector = YOLO(detector)
    return detector, classifier, database, class_dict, df
