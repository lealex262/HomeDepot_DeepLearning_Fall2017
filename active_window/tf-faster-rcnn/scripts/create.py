import os, shutil, random
from xml.dom import minidom
import untangle


# Relative Paths
build_folder_dir = "../data/Imagenet"
img_label_path = "../data/Image Labels"

#set up directory structure
if os.path.exists(build_folder_dir):
    shutil.rmtree(build_folder_dir + '/data/Annotations')
    shutil.rmtree(build_folder_dir + '/data/ImageSets')
    os.makedirs(build_folder_dir + '/data/Annotations')
    os.makedirs(build_folder_dir + '/data/ImageSets')
else:
    os.makedirs(build_folder_dir)
    os.makedirs(build_folder_dir + "/data") 
    os.makedirs(build_folder_dir + "/data/Annotations")
    os.makedirs(build_folder_dir + "/data/Images")
    os.makedirs(build_folder_dir + "/data/ImageSets")
img_classes = os.listdir(img_label_path)


# Creates dictionary to assign a class label to a class index. eg: {backhoe:0}
classes_file = open("predefined_classes.txt", "r")
img_Class_dictionary = {}
for index, img_class in enumerate(classes_file.read().splitlines()) :
    if(len(img_class) > 0):
        img_Class_dictionary[img_class] = index + 1
img_Class_dictionary["worker"] = img_Class_dictionary["person"]
img_Class_dictionary["lo"] = img_Class_dictionary["loader"]
img_Class_dictionary["fl"] = img_Class_dictionary["forklift"]
img_Class_dictionary["ex"] = img_Class_dictionary["excavator"]


#img parameters
#num_imgs_per_class = 200
train_percent = 0.8



#creates train and test files
train_imgs = []
test_imgs = []
for img_class in img_classes:
    imgs_in_class = os.listdir(img_label_path + "/"+ img_class)
    num_train_imgs = int(len(imgs_in_class) * train_percent)
    num_test_imgs = len(imgs_in_class) - num_train_imgs
    random.shuffle(imgs_in_class)
    for index, img_name in enumerate(imgs_in_class):
        img_name_without_ex = os.path.splitext(img_name)[0]
        if(index < num_train_imgs):
            train_imgs.append(img_name_without_ex)
        else:
            test_imgs.append(img_name_without_ex)

train_file = open(build_folder_dir + "/data/ImageSets/train.txt", "w")
random.shuffle(train_imgs)
for img_name in train_imgs:
    train_file.write(img_name+"\n")

random.shuffle(test_imgs)
test_file = open(build_folder_dir + "/data/ImageSets/test.txt", "w")
for img_name in test_imgs:
    test_file.write(img_name+"\n")


# creates annotation files
for img_class in img_classes:
    for label_file in os.listdir(img_label_path + "/" + img_class):
        xmldoc = untangle.parse(img_label_path + "/" + img_class + "/" + label_file)
        annotation_file = open(build_folder_dir + "/data/Annotations/" + os.path.splitext(label_file)[0] + ".txt","w")
        num_boxes = len(xmldoc.annotation) - 6
        if(num_boxes > 1):
            for i in range(0,len(xmldoc.annotation.object)):
                class_name = xmldoc.annotation.object[i].children[0].cdata
                class_index = img_Class_dictionary[class_name]
                x_min = xmldoc.annotation.object[i].bndbox.children[0].cdata
                y_min = xmldoc.annotation.object[i].bndbox.children[1].cdata
                x_max = xmldoc.annotation.object[i].bndbox.children[2].cdata
                y_max = xmldoc.annotation.object[i].bndbox.children[3].cdata
                annotation_file.write(str(class_index) + " " + x_min + " " + y_min + " " + x_max + " " + y_max + "\n")
        elif(num_boxes == 1):
            class_name = xmldoc.annotation.object.children[0].cdata
            class_index = img_Class_dictionary[class_name]
            x_min = xmldoc.annotation.object.bndbox.children[0].cdata
            y_min = xmldoc.annotation.object.bndbox.children[1].cdata
            x_max = xmldoc.annotation.object.bndbox.children[2].cdata
            y_max = xmldoc.annotation.object.bndbox.children[3].cdata
            annotation_file.write(str(class_index) + " " + x_min + " " + y_min + " " + x_max + " " + y_max + "\n")
        else:
            continue


