import torch
import numpy as np
from tools import harmonic_score_gzsl, normalized_accuracy_zsl
import torch.optim as optim
from art.attacks import FastGradientMethod, DeepFool, CarliniL2Method
from art.classifiers import PyTorchClassifier
import torchvision
import torch.nn as nn
from fullgraph import FullGraph
import time
from art.defences import SpatialSmoothing, TotalVarMin


def zsl_launch(dataloader, unseenVectors, criterion, params):
    if params["dataset"] == "CUB":
        from configs.config_CUB import MODEL_PATH, SMOOTHED_MODEL_PATH
    elif params["dataset"] == "AWA2":
        from configs.config_AWA2 import MODEL_PATH, SMOOTHED_MODEL_PATH
    elif params["dataset"] == "SUN":
        from configs.config_SUN import MODEL_PATH, SMOOTHED_MODEL_PATH

    resnet = torchvision.models.resnet101(pretrained=True).cuda()
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    if params["hasDefense"] and params["defense"] == "label_smooth":
        model_ale = torch.load(SMOOTHED_MODEL_PATH).cuda()
    else:
        model_ale = torch.load(MODEL_PATH).cuda()

    full_graph = FullGraph(feature_extractor, model_ale, unseenVectors).cuda()
    full_graph.eval()
    optimizer = optim.SGD(full_graph.parameters(), lr=0.01, momentum=0.5) # Placeholder  optimizer


    if params["dataset"] == "CUB":
        no_classes = 50
    elif params["dataset"] == "AWA2":
        no_classes = 10
    elif params["dataset"] == "SUN":
        no_classes = 72

    classifier = PyTorchClassifier(model=full_graph,  loss=criterion,
                                   optimizer=optimizer, input_shape=(1, 150, 150), nb_classes=no_classes)

    if params["attack"] == "fgsm":
        batch_size = 1
        attack = FastGradientMethod(classifier=classifier, eps=params["fgsm_params"]["epsilon"], batch_size= batch_size)
    elif params["attack"] == "deepfool":
        batch_size = 1
        attack = DeepFool(classifier, max_iter=params["deepfool_params"]["max_iter"],
                          epsilon= params["deepfool_params"]["epsilon"],
                          nb_grads= params["deepfool_params"]["nb_grads_zsl"], batch_size = batch_size)
    elif params["attack"] == "carlini_wagner":
        batch_size = params["batch_size"] if params["custom_collate"] else 1

        attack = CarliniL2Method(classifier, confidence= params["carliniwagner_params"]["confidence"],
                                              learning_rate= params["carliniwagner_params"]["learning_rate"],
                                              binary_search_steps= params["carliniwagner_params"]["binary_search_steps"],
                                              max_iter= params["carliniwagner_params"]["max_iter"],
                                              initial_const=params["carliniwagner_params"]["initial_const"],
                                              max_halving=params["carliniwagner_params"]["max_halving"],
                                              max_doubling=params["carliniwagner_params"]["max_doubling"], batch_size=batch_size)


    preds = []
    preds_defended = []

    adv_preds = []
    adv_preds_defended = []

    labels_ = []
    start= time.time()

    if params["hasDefense"]:
        if params["defense"] == "spatial_smooth":
            defense = SpatialSmoothing(window_size = params["ss_params"]["window_size"])
        elif params["defense"] == "totalvar":
            defense = TotalVarMin(max_iter =params["totalvar_params"]["max_iter"])

    for index, sample in enumerate(dataloader):
        img = sample[0].numpy()
        label = sample[1].numpy()

        if params["clean_results"]:
            if params["hasDefense"] and params["defense"] != "label_smooth":
                img_def,_ = defense(img)
                predictions_defended = classifier.predict(img_def, batch_size = batch_size)
                preds_defended.extend(np.argmax(predictions_defended, axis=1))
            predictions = classifier.predict(img, batch_size=batch_size)
            preds.extend(np.argmax(predictions, axis=1))

        img_perturbed = attack.generate(x=img)
        if params["hasDefense"] and params["defense"] != "label_smooth":
            img_perturbed_defended, _ = defense(img_perturbed)
            predictions_adv_defended = classifier.predict(img_perturbed_defended, batch_size = batch_size)
            adv_preds_defended.extend(np.argmax(predictions_adv_defended, axis=1))

        predictions_adv = classifier.predict(img_perturbed, batch_size=batch_size)
        adv_preds.extend(np.argmax(predictions_adv, axis=1))
        labels_.extend(label)

        if index % 1000 ==0:
            print(index, len(dataloader))

    end=time.time()

    labels_ = np.array(labels_)
    adv_preds = np.array(adv_preds)
    preds_defended = np.array(preds_defended)
    adv_preds_defended = np.array(adv_preds_defended)

    acc_adversarial  = normalized_accuracy_zsl(adv_preds, labels_)

    if params["clean_results"]:
        preds = np.array(preds)
        acc_original = normalized_accuracy_zsl(preds, labels_)
        print("ZSL Clean:", acc_original)

        if params["hasDefense"] and params["defense"] != "label_smooth":
            acc_only_defense = normalized_accuracy_zsl(preds_defended, labels_)
            print("ZSL Clean + defended:", acc_only_defense)


    print("ZSL Attacked:", acc_adversarial)

    if params["hasDefense"] and params["defense"] != "label_smooth":
        acc_attack_defend = normalized_accuracy_zsl(adv_preds_defended, labels_)
        print("ZSL attacked+defended:", acc_attack_defend)

    print(end-start , "seconds passed for ZSL.")


def gzsl_launch(dataloader_seen, dataloader_unseen, all_vectors, criterion, params):

    if params["dataset"] == "CUB":
        from configs.config_CUB import MODEL_PATH, SMOOTHED_MODEL_PATH
    elif params["dataset"] == "AWA2":
        from configs.config_AWA2 import MODEL_PATH, SMOOTHED_MODEL_PATH
    elif params["dataset"] == "SUN":
        from configs.config_SUN import MODEL_PATH, SMOOTHED_MODEL_PATH

    resnet = torchvision.models.resnet101(pretrained=True).cuda()
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    if params["hasDefense"] and params["defense"] == "label_smooth":
        model_ale = torch.load(SMOOTHED_MODEL_PATH).cuda()
    else:
        model_ale = torch.load(MODEL_PATH).cuda()

    full_graph = FullGraph(feature_extractor, model_ale, all_vectors).cuda()
    full_graph.eval()
    optimizer = optim.SGD(full_graph.parameters(), lr=0.01, momentum=0.5)

    if params["dataset"] == "CUB":
        no_classes = 200
    elif params["dataset"] == "AWA2":
        no_classes = 50
    elif params["dataset"] == "SUN":
        no_classes = 717

    classifier = PyTorchClassifier(model=full_graph, loss=criterion,
                                   optimizer=optimizer, input_shape=(1, 150, 150), nb_classes=no_classes)

    if params["attack"] == "fgsm":
        batch_size = 1
        attack = FastGradientMethod(classifier=classifier, eps=params["fgsm_params"]["epsilon"], batch_size= batch_size)

    elif params["attack"] == "deepfool":
        batch_size =  1
        attack = DeepFool(classifier, max_iter=params["deepfool_params"]["max_iter"],
                          epsilon= params["deepfool_params"]["epsilon"],
                          nb_grads= params["deepfool_params"]["nb_grads_gzsl"], batch_size = batch_size)

    elif params["attack"] == "carlini_wagner":
        batch_size = params["batch_size"] if params["custom_collate"] else 1
        attack = CarliniL2Method(classifier, confidence= params["carliniwagner_params"]["confidence"],
                                              learning_rate= params["carliniwagner_params"]["learning_rate"],
                                              binary_search_steps= params["carliniwagner_params"]["binary_search_steps"],
                                              max_iter= params["carliniwagner_params"]["max_iter"],
                                              initial_const=params["carliniwagner_params"]["initial_const"],
                                              max_halving=params["carliniwagner_params"]["max_halving"],
                                              max_doubling=params["carliniwagner_params"]["max_doubling"], batch_size=batch_size)

    preds_seen = []
    preds_seen_defended = []

    adv_preds_seen = []
    adv_preds_seen_defended = []
    labels_seen_ = []

    start= time.time()
    if params["hasDefense"]:
        if params["defense"] == "spatial_smooth":
            defense = SpatialSmoothing(window_size = params["ss_params"]["window_size"])
        elif params["defense"] == "totalvar":
            defense = TotalVarMin(max_iter =params["totalvar_params"]["max_iter"])

    for index, sample in enumerate(dataloader_seen):
        img = sample[0].numpy()
        label = sample[1].numpy()

        if params["clean_results"]:
            if params["hasDefense"] and params["defense"] != "label_smooth":
                img_def, _ = defense(img)
                predictions_defended = classifier.predict(img_def, batch_size = batch_size)
                preds_seen_defended.extend(np.argmax(predictions_defended, axis=1))
            predictions = classifier.predict(img, batch_size=batch_size)
            preds_seen.extend(np.argmax(predictions, axis=1))

        img_perturbed = attack.generate(x=img)
        if params["hasDefense"] and params["defense"] != "label_smooth":
            img_perturbed_defended, _ = defense(img_perturbed)
            predictions_adv_defended = classifier.predict(img_perturbed_defended, batch_size = batch_size)
            adv_preds_seen_defended.extend(np.argmax(predictions_adv_defended, axis=1))

        predictions_adv = classifier.predict(img_perturbed, batch_size = batch_size)
        adv_preds_seen.extend(np.argmax(predictions_adv, axis=1))
        labels_seen_.extend(label)


        if index % 1000 ==0:
            print(index, len(dataloader_seen))

    labels_seen_ = np.array(labels_seen_)
    adv_preds_seen = np.array(adv_preds_seen)
    adv_preds_seen_defended = np.array(adv_preds_seen_defended)
    uniq_labels_seen = np.unique(labels_seen_)

    adv_preds_unseen = []
    adv_preds_unseen_defended = []
    labels_unseen_ = []

    if params["clean_results"]:
        preds_unseen = []
        preds_seen = np.array(preds_seen)
        preds_unseen_defended = []
        preds_seen_defended = np.array(preds_seen_defended)

    for index, sample in enumerate(dataloader_unseen):
        img = sample[0].numpy()
        label = sample[1].numpy()

        if params["clean_results"]:
            if params["hasDefense"] and params["defense"] != "label_smooth":
                img_def, _ = defense(img)
                predictions_defended = classifier.predict(img_def, batch_size = batch_size)
                preds_unseen_defended.extend(np.argmax(predictions_defended, axis=1))
            predictions = classifier.predict(img, batch_size=batch_size)
            preds_unseen.extend(np.argmax(predictions, axis=1))

        img_perturbed = attack.generate(x=img)
        if params["hasDefense"] and params["defense"] != "label_smooth":
            img_perturbed_defended, _ = defense(img_perturbed)
            predictions_adv_defended = classifier.predict(img_perturbed_defended, batch_size=batch_size)
            adv_preds_unseen_defended.extend(np.argmax(predictions_adv_defended, axis=1))

        predictions_adv = classifier.predict(img_perturbed, batch_size=batch_size)
        adv_preds_unseen.extend(np.argmax(predictions_adv, axis=1))
        labels_unseen_.extend(label)


        if index % 1000 ==0:
            print(index, len(dataloader_unseen))

    end= time.time()


    labels_unseen_ = np.array(labels_unseen_)
    adv_preds_unseen = np.array(adv_preds_unseen)
    adv_preds_unseen_defended = np.array(adv_preds_unseen_defended)
    uniq_labels_unseen = np.unique(labels_unseen_)


    combined_labels = np.concatenate((labels_seen_, labels_unseen_))
    combined_preds_adv = np.concatenate((adv_preds_seen, adv_preds_unseen))
    combined_preds_adv_defended  = np.concatenate((adv_preds_seen_defended, adv_preds_unseen_defended))

    if params["clean_results"]:
        preds_unseen = np.array(preds_unseen)
        combined_preds = np.concatenate((preds_seen, preds_unseen))

        seen, unseen, h = harmonic_score_gzsl(combined_preds, combined_labels, uniq_labels_seen, uniq_labels_unseen)
        print("GZSL Clean (s/u/h):", seen, unseen, h)

        if params["hasDefense"] and params["defense"] != "label_smooth":
            preds_unseen_defended = np.array(preds_unseen_defended)
            combined_preds_defended = np.concatenate((preds_seen_defended, preds_unseen_defended))
            seen, unseen, h = harmonic_score_gzsl(combined_preds_defended, combined_labels, uniq_labels_seen, uniq_labels_unseen)
            print("GZSL Clean + defended (s/u/h):", seen, unseen, h)

    seen, unseen, h = harmonic_score_gzsl(combined_preds_adv, combined_labels, uniq_labels_seen, uniq_labels_unseen)
    print("GZSL Attacked (s/u/h):", seen, unseen, h)

    if params["hasDefense"] and params["defense"] != "label_smooth":
        seen, unseen, h = harmonic_score_gzsl(combined_preds_adv_defended, combined_labels, uniq_labels_seen, uniq_labels_unseen)
        print("GZSL Attacked + defended (s/u/h):", seen, unseen, h)


    print(end-start , "seconds passed for GZSL.")



