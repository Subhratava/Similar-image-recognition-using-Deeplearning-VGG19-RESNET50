import os
import numpy as np
import tensorflow as tf
#from sklearn.neighbors import NearestNeighbors
import skimage.io
from skimage.transform import resize
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.distance import cdist
import matplotlib.image as imgm
import matplotlib.pyplot as plt
import streamlit as st
from math import sqrt

# Read image
def read_img(filePath):
    return skimage.io.imread(filePath, as_gray=False)

# Read images with common extensions from a directory
def read_imgs_dir(dirPath, extensions):
    args = [os.path.join(dirPath, filename)
            for filename in os.listdir(dirPath)
            if any(filename.lower().endswith(ext) for ext in extensions)]

    imgs = [read_img(arg) for arg in args]
    return imgs,args

# Normalize image data [0, 255] -> [0.0, 1.0]
def normalize_img(img):
    return img / 255.

# Resize image
def resize_img(img, shape_resized):
    img_resized = resize(img, shape_resized,
                         anti_aliasing=False,
                         preserve_range=True)
    assert img_resized.shape == shape_resized
    return img_resized

def get_folder():
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        file_path = filedialog.askdirectory()
        root.destroy() 
        return str(file_path)


def model_vgg(h,w,imgs_train,dir,n):
    extensions = [".jpg", ".jpeg",".png"]
    print(imgs_train.shape)
    imgs_train = resize_img(imgs_train, (h,w,3))
    img_temp = imgs_train
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                        input_shape=imgs_train.shape)
    #model.summary()
    shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
    input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
    n_epochs = None
    img_transformed = []
    img_temp = normalize_img(img_temp)
    img_transformed.append(img_temp)
    X_train = np.array(img_transformed).reshape((-1,) + input_shape_model)
    E_train = model.predict(X_train)
    E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))

    imgs_test , path_test = read_imgs_dir(dir, extensions)
    #print(len(path_test))
    sims = []
    for i in range(len(imgs_test)):
        img = imgs_test[i]
        img_transformed_test = []
        img_temp = resize_img(img,shape_img_resize)
        img_temp = normalize_img(img_temp)
        #st.image(img_temp, caption=path_test[i])
        img_transformed_test.append(img_temp)
        X_test = np.array(img_transformed_test).reshape((-1,) + input_shape_model)
        #print(input_shape_model," ",i)
        E_test = model.predict(X_test)
        E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model))) #final vector
        sims.append((1 - cdist(E_train_flatten, E_test_flatten, 'cosine')[0][0],path_test[i]))
        #print(path_test[i], " ",(1 - cdist(E_train_flatten, E_test_flatten, 'cosine')[0][0]))
        i += 1
    st.write('#### Possible similar images.')
    res = sorted(sims, key=lambda tup: tup[0], reverse=True)
    rows = int(sqrt(n)) + 1
    cols = int(sqrt(n)) + 1
    axes = []
    print("done")
    fig=plt.figure(figsize=(n,n))
    i = 0
    n = min(len(res),n)
    for i in range(n):
        b = imgm.imread(res[i][1])
        print(res[i][1])
        axes.append(fig.add_subplot(rows,cols,i+1))
        plt.imshow(b)
        i+=1
    fig.tight_layout()
    #plt.show()
    st.pyplot(fig)



    


