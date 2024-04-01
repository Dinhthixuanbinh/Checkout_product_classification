from tools.create_database import create_module
from tools.process_output import get_table, plot_bbox
from predict import inferecne_img
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import os


@st.cache_resource()
def get_detector(path=r'best.pt'):
    model = YOLO(path)
    return model


@st.cache_data()
def get_data(_detector):
    return create_module(directory_path=r'label_crop_top10',
                         id_class_path=r'map_id.csv',
                         detector=_detector,
                         dataframe=r'class_names_with_index.csv',
                         state=False)


def main():
    detector = get_detector()
    result = get_data(detector)

    st.title("Automated Retail Checkout")
    uploaded_image = st.file_uploader("Choose an image...",
                                      type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)  # Create two columns

    with col1:

        if uploaded_image is not None:
            pil_image = Image.open(uploaded_image)
            image_path = "uploaded_image.jpg"
            pil_image.save(image_path)
            products, predictions = inferecne_img(img_path=image_path,
                                                  embedding_model=result[1],
                                                  id_class_dict=result[3],
                                                  database=result[2],
                                                  dataframe=result[-1],
                                                  dectector=result[0])

            out_img = plot_bbox(image_path, predictions, state=False)
            st.image(out_img, caption="Uploaded Image", use_column_width=True)
            os.remove(image_path)

    with col2:
        if uploaded_image is not None:
            st.table(get_table(products))


if __name__ == "__main__":
    main()
