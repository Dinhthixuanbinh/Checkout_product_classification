from collections import Counter
import pandas as pd
import cv2


def calculate_acc(lst):
    total_elements = len(lst)
    if total_elements == 0:
        return 0  # Return 0 if the list is empty
    count_ones = lst.count(1)
    ratio_ones = count_ones / total_elements
    return ratio_ones


def model_pred(embedding_model, cropped_img,
               class_id, database,
               id_class_dict,
               state=False):

    embedding_vect = embedding_model.get_vec(cropped_img)
    if state:
        class_name = class_id
    else:
        class_name = id_class_dict.get(int(class_id))
    index, label_list = database.get(class_name)
    _, indexs = index.search(embedding_vect, 1)
    return label_list[indexs[0][0]]


def get_name_id(dataframe, number_id, state=True):
    df = dataframe
    if state:
        df = pd.read_csv(dataframe)
    return df[df.index == number_id].class_name.values[0]


def plot_bbox(image_path, predictions,
              target_size=(640, 640),
              state=True):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, target_size)

    # Iterate over predictions and draw bounding boxes
    for prediction in predictions:
        label, (left, top, right, bottom) = prediction
        last_underscore_index = label.rfind('_')
        color = int(label[last_underscore_index + 1:])

        left = int(left * target_size[0] / img.shape[1])
        top = int(top * target_size[1] / img.shape[0])
        right = int(right * target_size[0] / img.shape[1])
        bottom = int(bottom * target_size[1] / img.shape[0])

        cv2.rectangle(resized_img, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        # Put label text
        cv2.putText(resized_img, label, (left, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, color, 50+color), 2)
    if state:
        # Display the resized image with bounding boxes
        cv2.imshow('Bounding Boxes', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


def format_lst(lst):
    lst = Counter(lst)
    # Print the counts
    for element, count in lst.items():
        print(f"Product {element}: {count}")


def get_table(lst):
    lst = Counter(lst)
    df = pd.DataFrame(lst.items(), columns=['Item', 'Quantity'])
    return df.sort_values(by='Quantity', ascending=False)
