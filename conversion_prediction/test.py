from dataset import Dataset


dataset = Dataset("train.json")
batch_size = 20

feature, label = dataset.next_batch(batch_size, 'train')