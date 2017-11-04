import json
from pprint import pprint
from collections import namedtuple

class Dataset:

    def __init__(self, jsonfile):
        with open(jsonfile) as data_file:
            data = json.load(data_file)

        self.features = []
        self.isSale = []

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

            self.features.append(item_feature)
            self.isSale.append(int(item["isSale"]))

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        print(len(self.features))
        self.train_size = len(self.features)
        self.test_size = len(self.isSale)
        self.n_classes = 1

        def next_batch(self, batch_size, phrase):
            # Get next batch of image (path) and labels
            if phase == 'train':
                if self.train_ptr < self.train_size:
                    session = self.features[self.train_ptr:self.train_ptr + batch_size]
                    labels = self.isSale[self.train_ptr:self.train_ptr + batch_size]
                    self.train_ptr += batch_size
                else:
                    new_ptr = (self.train_ptr + batch_size) % self.train_size
                    session = self.features[self.train_ptr:] + self.features[:new_ptr]
                    labels = self.isSale[self.train_ptr:] + self.isSale[:new_ptr]
                    self.train_ptr = new_ptr
            elif phase == 'test':
                if self.test_ptr + batch_size < self.test_size:
                    session = self.features[self.train_ptr:self.train_ptr + batch_size]
                    labels = self.isSale[self.train_ptr:self.train_ptr + batch_size]
                    self.train_ptr += batch_size
                else:
                    new_ptr = (self.train_ptr + batch_size) % self.train_size
                    session = self.features[self.train_ptr:] + self.features[:new_ptr]
                    labels = self.isSale[self.train_ptr:] + self.isSale[:new_ptr]
                    self.train_ptr = new_ptr
            else:
                return None, None

            return session, labels



