import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def processed_for_test(image_path, model):
    img = tf.keras.utils.load_img(image_path)
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.resize(img, (224, 224))

    input_arr = tf.keras.applications.resnet_v2.preprocess_input(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return img/255.0, predictions


def test_result(filename, result):
    # result is an embedding vector of shape (1,32)
    # return False if 2 out of 3 vector return result>0.5
    anchors = np.loadtxt(filename, delimiter=',')
    dist = anchors-result
    dist = np.square(dist)
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    dist_05 = np.less(dist, 0.5)
    dist_06 = np.less(dist, 0.6)
    if np.sum(dist_06) >= 2 or np.sum(dist_05) >= 1:
        return True
    else:
        return False


def most_similar(filename, result):
    anchors = np.loadtxt(filename, delimiter=',')
    dist = anchors-result
    dist = np.square(dist)
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return np.sum(dist)


def verify(img_path, model_version='03_60', model_path=None, db_fol=None):
    if model_path is None:
        if model_version in ['05_60', '00_60', '03_60', '03_90']:
            model_path = f'model/model_{model_version}.h5'
            model = tf.keras.models.load_model(model_path)
            filedirs = [os.path.join(f'emb_db/{model_version}', filename)
                        for filename in os.listdir(f'emb_db/{model_version}')]
        else:
            st.write(
                'Please specify a model version or to provide the path to the model')
            return None
    else:
        if db_fol is None:
            st.write('Please provide the path to the embedding vector folder')
            return None
        else:
            model_path = os.path.normpath(model_path)
            model = tf.keras.models.load_model(model_path)
            filedirs = [os.path.join(db_fol, fname)
                        for fname in os.listdir(db_fol)]

    img, test_emb = processed_for_test(img_path, model)
    min_dist = 10
    chosen = None
    existed = False

    for file in filedirs:
        if test_result(file, test_emb) == 1:
            existed = True
            chosen = file

    if not existed:
        for file in filedirs:
            if most_similar(file, test_emb) <= min_dist:
                min_dist = most_similar(file, test_emb)
                chosen = file
    chosen = os.path.normpath(chosen)
    fname = chosen.split('\\')[-1].split('.')[0]
    chosen = os.path.join('img_db', fname)

    return existed, chosen


def main():
    # Set page configs. Get emoji names from WebFx
    st.set_page_config(page_title="Nhom 5 - Style Transfer", layout="centered")
    title1 = '<p style="text-align: center;font-size: 50px;font-weight: 350;font-family:Monospace "> PROJECT CUỐI KÌ - NHÓM 5 </p>'
    title2 = '<p style="text-align: center;font-size: 30px;font-weight: 350;font-family:Monospace "> Phát triển phần mềm nâng cao</p>'
    title3 = '<p style="text-align: center;font-size: 30px;font-weight: 350;font-family:Monospace "> cho tính toán khoa học</p>'
    title4 = '<p style="text-align: center;font-size: 15px;font-weight: 350;font-family:Monospace "> Trần Hoàng Đức - Lê Thị Kim Ngân - Nguyễn Trung Anh - Trần Bảo Trung  </p>'
    st.markdown(title1, unsafe_allow_html=True)
    st.markdown(title2, unsafe_allow_html=True)
    st.markdown(title3, unsafe_allow_html=True)
    st.markdown(title4, unsafe_allow_html=True)
    st.markdown(
        "<p  style='text-align: center;font-size: 25px;'><b> <i> NHẬN DẠNG MẶT MÈO </i> </b></p>", unsafe_allow_html=True
    )
    st.image(image="./assets/banner.jpg")
    st.markdown("</br>", unsafe_allow_html=True)
    st.markdown(
        "<p  style='font-size: 25px;'><b> <i> 1. GIỚI THIỆU VỀ WEB: </i> </b></p> Sử dụng mô hình gốc là mạng Resnet, kỹ thuật transfer learning với hàm mất mát là triplet loss. "
        "Hệ thống sẽ so sánh hình được upload với dữ liệu trong database xác định con mèo có trong database hay không, "
        "và trả ra các hình ảnh khác của nó nếu có, hoặc trả ra hình mèo tương tự nhất.", unsafe_allow_html=True
    )
    st.markdown("</br>", unsafe_allow_html=True)
    st.markdown(
        "<p  style='font-size: 25px;'><b> <i> 2. TÌM MÈO CỦA BẠN: </i> </b></p>", unsafe_allow_html=True
    )
    st.markdown("</br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.write("Uploaded Image")
        image = Image.open(uploaded_file)
        existed, chosen_fol = verify(uploaded_file)
        img_list = [os.path.join(chosen_fol, fname)
                    for fname in os.listdir(chosen_fol)]
        size = len(img_list)-1
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(image, width=300)
        if existed:
            st.markdown("<p  style='text-align: center;font-size: 20px;'><b> <i> Ảnh mèo này đã tồn tại trong cơ sở dữ liệu </i> </b></p>", unsafe_allow_html=True)
            st.write(f'The cat is already stored in the database')
            another_cat_button = st.button(
                "Bạn có muốn xem ảnh của chú mèo khác cùng loại?")
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                if another_cat_button:
                    st.image(Image.open(
                        img_list[random.randint(0, size)]), width=300)
        else:
            st.markdown("<p  style='text-align: center;font-size: 20px;'><b> <i> Ảnh mèo này chưa tồn tại trong cơ sở dữ liệu </i> </b></p>", unsafe_allow_html=True)
            # st.write('The cat is not yet recorded in our database!')
            another_cat_button = st.button(
                "Bạn có muốn tìm kiếm ảnh của chú mèo khác gần giống với mèo của bạn?")
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                if another_cat_button:
                    st.image(Image.open(
                        img_list[random.randint(0, size)]), width=300)


if __name__ == "__main__":
    main()
