# Dataset path
# create_data_sets()
train_list = 'train.txt'
test_list = 'test.txt'

# Network params
n_classes = 6
keep_rate = 0.5

# Graph input
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_var = tf.placeholder(tf.float32)

# Model
vgg = VGG16(x)
pred = vgg.getVGG16()

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

# Launch the graph
with tf.Session() as sess:
    print('Init variable')
    sess.run(init)

    # Load weights for VGG
    vgg.load_weights("vgg16_weights.npz", sess)

    saver.restore(sess, "model_files/model_final.ckpt")
    results = open("results.txt", "w+")
    for i in range(int((dataset.final_test_size)/batch_size)):
        batch_tx, paths = dataset.next_batch_test(batch_size, 'test')
        pred_label = sess.run((tf.argmax(pred, 1)), feed_dict={x: batch_tx})
        for j in range(len(pred_label)):
            results.write(labels_dict[str(j)] + "|" + paths[j])
    results.close()