from torch.utils.data import Dataset
from PIL import Image
import os
import torch

label_map = {
    0:"antelope",29: "bat",3 : " beaver",8 : " blue+whale",40 : " bobcat",36 : " buffalo",32 : " chihuahua",
    24 : " chimpanzee",45 : " collie",48 : " cow",4  : "dalmatian",39 : " deer", 49 : " dolphin",
    18 : " elephant",21 : " fox",7 : " german+shepherd",38 : "giant+panda",30 : "giraffe",
    19 : "gorilla",1 : "grizzly+bear",25 : "hamster",13 : "hippopotamus", 6 : "horse", 17 : "humpback+whale",
    2 : "killer+whale",14 : "leopard",42 : "lion",11 : "mole",15 : "moose",43 : "mouse", 35 : "otter",
    20 : "ox",5 : "persian+cat", 41 : "pig",44 : "polar+bear",28 : "rabbit", 47 : "raccoon", 33 : "rat",27 : "rhinoceros",
    23 : "seal",22 : "sheep", 9 : "siamese+cat", 10 : "skunk", 16 : "spider+monkey", 26 : "squirrel",
    12 : "tiger", 46 : "walrus",34: "weasel",31: "wolf", 37:"zebra"
}

zsl_to_gzsl_label_indexes= {0: 6, 1: 8, 2: 22, 3:23,
4: 29, 5: 30, 6: 33, 7:40,
8: 46, 9: 49
}

unseen_indices = [6 ,8,22,23,29,30,33,40,46,49]
seen_indices = [x for x in range(len(label_map)) if x not in unseen_indices]

class AWADataset(Dataset):
    def __init__(self, indexes, files, labels , data_root, zsl = False, transform=None):
        self.index_instances = indexes
        self.data_root = data_root

        self.file_names = files[self.index_instances - 1]
        self.labels = (labels[self.index_instances - 1] -1)

        if zsl:
            self.map_labels_zsl()

        self.transform = transform

    def __len__(self):
        return len(self.index_instances)


    def map_labels_zsl(self):
        for index, i in enumerate(self.labels):
            if i == 6:
                self.labels[index] = 0
            elif i == 8:
                self.labels[index] = 1
            elif i == 22:
                self.labels[index] = 2
            elif i == 23:
                self.labels[index] = 3
            elif i == 29:
                self.labels[index] = 4
            elif i == 30:
                self.labels[index] = 5
            elif i == 33:
                self.labels[index] = 6
            elif i == 40:
                self.labels[index] = 7
            elif i == 46:
                self.labels[index] = 8
            elif i == 49:
                self.labels[index] = 9
            else:
                print("Wrong label for zsl conversion!", i)


    def fetch_batch(self, idx):
        im_name = self.file_names[idx][0][0][0].split('JPEGImages/')[1]
        image_file = os.path.join(self.data_root, im_name)

        img_pil = Image.open(image_file).convert("RGB")
        label = self.labels[idx]
        label = label[0]
        label = torch.Tensor(label)
        img_tensor = self.transform(img_pil)

        return (img_tensor, label)

    def __getitem__(self, idx):
        batch = self.fetch_batch(idx)

        return batch

