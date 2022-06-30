from asyncio.windows_events import NULL
from logging import PlaceHolder
import os
from tkinter import Label
import numpy as np
import tensorflow as tf
#from sklearn.neighbors import NearestNeighbors
import skimage.io
import streamlit as st
from PIL import Image, ImageOps
from engine import resize_img, get_folder, model_vgg


def main():
    h = 224
    w = 224
    st.sidebar.write('#### Upload an image to search.')
    uploaded_file = st.sidebar.file_uploader('',
                                             type=['png', 'jpg', 'jpeg'],
                                             accept_multiple_files=False)

    st.sidebar.write('#### Select search folder.')
    direc = ''
    PlaceHolder = st.sidebar.empty()
    try:
        direc = st.session_state.catch_rand
        PlaceHolder.text_area(label="Directory", value=direc)
    except:
        print("continuing")

    browse = st.sidebar.button("Browse")
    if browse:
        direc = get_folder()
        st.session_state.catch_rand = direc
        PlaceHolder.text_area(label="Directory", value=direc,key = 1)

    st.title("Find similar images on your device!")
    num_img = st.sidebar.number_input(
        "Enter no of images to be mapped", min_value=2, max_value=20)
    gpu_option = st.sidebar.checkbox("GPU Acceleration", value=True)
    if gpu_option == False:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if gpu_option == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with st.sidebar.form(key='columns_in_form'):
        c1, c2 = st.columns(2)
        with c1:
            h = int(st.text_input("Image height", value=h))
        with c2:
            w = int(st.text_input("Image width", value=w))

        submitButton = st.form_submit_button(label='Resize & Proceed')

    model_option = st.sidebar.selectbox(
        "Select Model", options=("VGG19", "Resnet50"), index=0)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        imgs_train = (np.array(image))
        st.image(imgs_train, caption="Uploaded Image", width=512)
        # print(imgs_train.shape)
        if submitButton:
            print(direc)
            if model_option == "VGG19":
                print(num_img)
                model_vgg(h, w, imgs_train, direc, int(num_img))


if __name__ == "__main__":
    main()
