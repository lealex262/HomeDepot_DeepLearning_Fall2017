import json
from pprint import pprint
from collections import namedtuple

class Dataset:

    def __init__(self, jsonfile):
        with open(jsonfile) as data_file:
            data = json.load(data_file)

        features = []
        isSale = []

        for item in data:
            item_feature = []
            dateTime = item["startLocalDateTimeString"].split("T")
            date = dateTime[0].split("-")
            item_feature.append(int(date[0])) #year
            item_feature.append(int(date[1])) #month
            item_feature.append(int(date[2])) #day
            time = dateTime[1].split(":")
            item_feature.append(int(time[0])) #hour
            item_feature.append(int(time[1])) #min
            item_feature.append(int(time[2])) #sec
            item_feature.append(item["startUnixTimeSecondsGMT"])
            item_feature.append(item["endUnixTimeSecondsGMT"])
            item_feature.append(item["numClickstreams"])
            item_feature.append(item["numSearches"])
            item_feature.append(item["numNoResultsFound"])
            item_feature.append(item["numAutoCompleteClicks"])
            item_feature.append(item["numRelatedTermClicks"])
            item_feature.append(item["numSuggestedProductClicks"])
            item_feature.append(int(item["isMobileVisit"]))

            features.append(item_feature)
            isSale.append(int(item["isSale"]))

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.data_split = .8
        self.train_size = int(len(features) * self.data_split)
        self.test_size = len(features) - self.train_size
        self.n_classes = 1

        # Create training and test sets
        print(self.train_size)
        self.test_label = isSale[self.train_size:len(isSale)]
        self.test_features = features[self.train_size:len(isSale)]
        self.train_label = isSale[0:self.train_size]
        self.train_features = features[0:self.train_size]


    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr < self.train_size:
                session = self.train_features[self.train_ptr:self.train_ptr + batch_size]
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                session = self.train_features[self.train_ptr:] + self.train_features[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                session = self.test_features[self.test_ptr:self.test_ptr + batch_size]
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                session = self.test_features[self.test_ptr:] + self.test_features[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None
        
        return session, labels



