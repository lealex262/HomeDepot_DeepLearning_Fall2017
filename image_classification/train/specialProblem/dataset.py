import numpy as np
from scipy.misc import imread, imresize
import cv2

class Dataset:
    def __init__(self, train_list, test_list):
        # Load training images (path) and labels
        with open(train_list) as f:
            lines = f.readlines()
            self.train_image = []
            self.train_label = []
            for l in lines:
                items = l.split()
                self.train_image.append(items[0])
                self.train_label.append(int(items[1]))

        # Load testing images (path) and labels
        with open(test_list) as f:
            lines = f.readlines()
            self.test_image = []
            self.test_label = []
            for l in lines:
                items = l.split()
                self.test_image.append(items[0])
                self.test_label.append(int(items[1]))

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.crop_size = 224
        self.scale_size = 256
        # alexnet mean
        # self.mean = np.array([104., 117., 124.])
        #vgg mean
        self.mean = np.array([123.68, 116.779, 103.939])
        self.n_classes = 6

    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr  < self.train_size:
                paths = self.train_image[self.train_ptr:self.train_ptr + batch_size]
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                paths = self.train_image[self.train_ptr:] + self.train_image[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None

        # Read images
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for i in range(len(paths)):
            if(i > 9):
                print(len(paths))
                print(paths)
                continue
            img = cv2.imread(paths[i])
            img = cv2.resize(img, (self.scale_size, self.scale_size))

            if(len(img.shape) ==  3):
                if (img.shape[2] == 3):
                    img = img.astype(np.float32)
                    img -= self.mean
                    #print(img.shape)
                    shift = int((self.scale_size-self.crop_size)/2)
                    img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
                    images[i] = img_crop

        # Expand labels
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            if (i > 9):
                continue
            one_hot_labels[i][labels[i]] = 1
        return images, one_hot_labels

