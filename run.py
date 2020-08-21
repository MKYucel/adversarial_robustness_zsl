import numpy as np
import torch
from dataloaders.awa_dataloader import AWADataset
from dataloaders.cub_dataloader import CUBDataset
from dataloaders.sun_dataloader import SunDataset
import scipy.io as sio
from torchvision import transforms
from torch.utils.data import DataLoader
from tools import load_data, load_json
import sys
from attack_defend_factory import zsl_launch, gzsl_launch
import argparse

def main():
    parser = argparse.ArgumentParser(description='JSON file')
    parser.add_argument("--path", dest="json_path", type=str, help='path to json file. defaults to params.json', default= "params.json")
    args = parser.parse_args()
    print(args.json_path)
    params = load_json(args.json_path)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if params["dataset"] == "CUB":
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize
        ])
        from configs.config_CUB import MAIN_DATAPATH, TEST_DATAPATH
        att_split = sio.loadmat(params["CUB_paths"]["att_split"])
        root = params["CUB_paths"]["root"]
        metaData = sio.loadmat(params["CUB_paths"]["metaData"])
        print("CUB Dataset chosen.")
        dataloader_placeholder = CUBDataset

    elif params["dataset"] == "AWA2":
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        from configs.config_AWA2 import MAIN_DATAPATH, TEST_DATAPATH
        att_split = sio.loadmat(params["AWA2_paths"]["att_split"])
        root = params["AWA2_paths"]["root"]
        metaData = sio.loadmat(params["AWA2_paths"]["metaData"])
        dataloader_placeholder = AWADataset
        print("AWA2 Dataset chosen.")

    elif params["dataset"] == "SUN":
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize
        ])
        from configs.config_SUN import MAIN_DATAPATH, TEST_DATAPATH
        att_split = sio.loadmat(params["SUN_paths"]["att_split"])
        root = params["SUN_paths"]["root"]
        metaData = sio.loadmat(params["SUN_paths"]["metaData"])
        print("SUN Dataset chosen.")
        dataloader_placeholder = SunDataset

    else:
        print("Invalid dataset chosen. ")
        sys.exit()

    all_class_vector = load_data(MAIN_DATAPATH + 'all_class_vec.mat', "all_class_vec")
    train_val_labels = load_data(TEST_DATAPATH + 'trainval_labels.mat', 'trainval_labels')
    unseen_labels =load_data(TEST_DATAPATH + 'test_unseen_labels.mat','test_unseen_labels')

    seenClassIndices    = np.unique(train_val_labels)
    unseenClassIndices  = np.unique(unseen_labels)


    unseenVectors       = torch.from_numpy(all_class_vector[unseenClassIndices, :]).float().cuda()
    allVectors          = torch.from_numpy(all_class_vector).float().cuda()



    trainval_indexes = att_split["trainval_loc"]
    test_unseen_indexes = att_split["test_unseen_loc"]
    test_seen_indexes = att_split["test_seen_loc"]

    files = metaData["image_files"]
    labels = metaData["labels"]


    dataloader_zsl = DataLoader(dataloader_placeholder(test_unseen_indexes, files, labels, root, zsl= True,  transform=preprocess), batch_size=1,
                            shuffle=params["shuffle_dataset"],num_workers=params["num_workers"], pin_memory=params["pin_memory"])

    dataloader_unseen = DataLoader(dataloader_placeholder(test_unseen_indexes, files, labels, root, transform=preprocess), batch_size=1,
                            shuffle=params["shuffle_dataset"],num_workers=params["num_workers"], pin_memory=params["pin_memory"])
    dataloader_seen = DataLoader(dataloader_placeholder(test_seen_indexes, files, labels, root, transform=preprocess), batch_size=1,
                            shuffle=params["shuffle_dataset"],num_workers=params["num_workers"], pin_memory=params["pin_memory"])



    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    if params["zsl_test"]:
        zsl_launch(dataloader_zsl, unseenVectors, criterion, params)

    print("-------" * 20)
    if params["gzsl_test"]:
        gzsl_launch(dataloader_seen, dataloader_unseen, allVectors, criterion, params)

if __name__ == '__main__':
    main()



