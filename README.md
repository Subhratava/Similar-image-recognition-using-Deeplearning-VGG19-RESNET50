# Similar-image-recognition-using-Deeplearning-VGG19-RESNET50
A demo application to showcase the use of transfer learning to recognize similar images on your storage in a given specified folder.

# Libraries used
1. Numpy.
2. Tensorflow backend for keras.
3. PIL (image library for python).
4. Streamlit for ui development

# Detailed description and workflow
Used a pretrained model (VGG19 currently , RESNET50 and inseptionv3 will be added later) and combined with cosine similarity of vectors ,to match similar looking images on the user hard drive, could be used to find out which images are looking similar to each other.

# How to run

1. Open cmd or powershell or any other command line interface to run streamlit with command ```streamlit run ui.py``` .
2. Input the image for which you like to find the matches.
3. Input search directory.
4. Resize according to your wish (higher resolution will require higher VRAM).
5. Proceed

# Result should look similar to this output.

![project_result](https://user-images.githubusercontent.com/41603518/176736570-f3b3a512-6d59-4e5f-af73-ffaa199cdf42.jpg)
