import tensorflow as tf
import numpy as np
import sys
sys.path.append("../scripts")
from model import Model
from VGG16 import VGG16
from dataset import Dataset
from network import *
from datetime import datetime
from createdatasets import create_data_sets


def main():

    # Dataset path
    # create_data_sets()
    train_list = 'train.txt'
    test_list = 'test.txt'

    # Learning params
    learning_rate = 0.0005
    training_iters = 7000 # 10 epochs
    batch_size = 10
    display_step = 20
    test_step = 100 # 0.5 epoch
    save_step = 1000

    # Network params
    n_classes = 6
    keep_rate = 0.5

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    # pred = Model.alexnet(x, keep_var)

    vgg = VGG16(x)
    pred = vgg.getVGG16()


    # pred = resnet({'data': x}).layers["fc1000"]


    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Init
    init = tf.initialize_all_variables()

    # Load dataset
    dataset = Dataset(train_list, test_list)

    # create a saver
    saver = tf.train.Saver()

    #Create log of the results in a text file
    #textFilesPath = "../specialProblem/"
    #log = open(textFilesPath + "log.txt", "a")
    #sys.stdout = log

    #Create Confusion matrix
    confusionMatrix = np.zeros((6, 6))
    confusionTotal = [0, 0, 0, 0, 0, 0]

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)

        # Load pretrained model

        # Load weights for AlexNet
        # load_with_skip('caffenet.npy', sess, ['fc8']) # Skip weights from fc8

        # Load weights for VGG
        vgg.load_weights("vgg16_weights.npz", sess)

        print('Start training')
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})
            # save mid-point models
            if step % save_step == 0:
                save_path = saver.save(sess,"model_files/model_" + str(step) +  ".ckpt")
                print("model saved in file: ", save_path)

            # Display testing status
            if step%test_step == 0:
                test_acc = 0.
                test_count = 0
                # confusionMatrix = np.zeros((6, 6))
                # confusionTotal = [0, 0, 0, 0, 0, 0]
                # validPredLabel_count = 0

                for _ in range(int(dataset.test_size/batch_size)):
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                    acc, pred_label, actual_label = sess.run((accuracy, tf.argmax(pred, 1), tf.argmax(y, 1)), feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1

                    #print(len(pred_label))

                    # if (max(pred_label) >= 6):
                    #     # print(batch_tx)
                    #     # print(actual_label)
                    #     # print(pred_label)
                    #     continue
                    # validPredLabel_count += 1
                    # i = 0
                    # while i < len(pred_label):
                    #         confusionMatrix[pred_label[i]][actual_label[i]] += 1
                    #         confusionTotal[actual_label[i]] += 1
                    #         i += 1

                test_acc /= test_count
                print( sys.stderr, "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))
                # print(validPredLabel_count)
                # for row in range(6):
                #     for col in range(6):
                #         confusionMatrix[row][col] = round((confusionMatrix[row][col] / validPredLabel_count), 2)
                # print(confusionMatrix)




            # Display training status
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                print( sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc))

            step += 1

        print("Finish!")
        # for x in range(10):
        #     for y in range(10):
        #         confusionMatrix[x][y] = confusionMatrix[x][y] / 120
        # print(confusionMatrix)
        #log.close()

if __name__ == '__main__':
    main()
