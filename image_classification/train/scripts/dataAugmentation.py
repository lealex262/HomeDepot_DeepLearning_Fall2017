import os
import numpy as np
import shutil

from PIL import Image
from scipy.misc import imread, imresize, pilutil
import cv2

def augment_data():
    imagesPath = "../../images/from_imageNet/"
    trainFilesPath = "../../images/trainingImg/"

    # Set parameters for data argumentation
    crop_size = 227
    scale_size = 256
    train_scale_size = 30;

    # Clear Training Folder
    if os.path.exists(trainFilesPath):
        shutil.rmtree(trainFilesPath)

    # create directories if they dont exist
    if not os.path.exists(trainFilesPath):
        os.makedirs(trainFilesPath)
    for folderName in os.listdir(imagesPath):
        if not os.path.exists(trainFilesPath + "/" + folderName):
            os.makedirs(trainFilesPath + "/" + folderName)

    for folderName in os.listdir(imagesPath):
        currPath = imagesPath.__add__("/%s" % folderName)
        for file in os.listdir(currPath):
            file_path = currPath + "/" + file
            filename = os.path.splitext(file)[0]

            # checks if image may be in an unsupported format and converts to jpg and deletes previous version of image
            try:
                img = cv2.imread(file_path)
                img = cv2.resize(img, (scale_size, scale_size))
            except Exception as e:
                print(file_path)
                new_file_path = currPath + "/" + filename + ".jpg"
                Image.open(file_path).convert('RGB').save(new_file_path)
                os.replace(file_path, new_file_path)
                img = cv2.imread(new_file_path)
                img = cv2.resize(img, (scale_size, scale_size))



            # Create copy of image but flipped
            flip_img = img.copy()
            flip_img = cv2.flip(flip_img, 1)

            # Enlarging image size and then cropping into 4 pieces
            img_scaled = cv2.resize(img, ((scale_size + train_scale_size), (scale_size + train_scale_size)))
            img_top_left_crop = img_scaled[:scale_size, :scale_size]
            img_top_right_crop = img_scaled[:scale_size, train_scale_size:scale_size + train_scale_size]
            img_bot_left_Crop = img_scaled[train_scale_size:scale_size + train_scale_size, :scale_size]
            img_bot_right_crop = img_scaled[train_scale_size:scale_size + train_scale_size,
                                 train_scale_size:scale_size + train_scale_size]

            save_file_path = trainFilesPath + folderName

            # saving the new images in the training folder
            cv2.imwrite(save_file_path + "/" + filename + "f.jpg", flip_img)
            cv2.imwrite(save_file_path + "/" + filename + "tl.jpg", img_top_left_crop)
            cv2.imwrite(save_file_path + "/" + filename + "tr.jpg", img_top_right_crop)
            cv2.imwrite(save_file_path + "/" + filename + "bl.jpg", img_bot_left_Crop)
            cv2.imwrite(save_file_path + "/" + filename + "br.jpg", img_bot_right_crop)



augment_data()