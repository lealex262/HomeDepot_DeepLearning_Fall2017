import re
data_file = "train_coords.txt"
text_file_path = "tf-faster-rcnn/data/Imagenet/data/ImageSets/"
annotation_file_path = "tf-faster-rcnn/data/Imagenet/data/Annotations/"
file = open(data_file)
data = file.read().splitlines()

print(len(data))
train_num = int(0.8 * len(data))

train_file = open(text_file_path + "train.txt", "w")
test_file = open(text_file_path +"test.txt", "w")

for i, line in enumerate(data):
    values = re.findall(r"[\w']+", line)
    tlx = values[0]
    tly = values[1]
    brx = values[2]
    bry = values[3]
    outputline = "1 " + tlx + " " + bry + " " + brx + " " + tly + "\n"
    annotation_file = open(annotation_file_path + "screenshot"+str(i+1)+".txt", "w")
    annotation_file.write(outputline)
    annotation_file.close()
    if i < train_num:
        train_file.write("screenshot" + str(i+1) + "\n")
    else:
        test_file.write("screenshot" + str(i+1) + "\n")
