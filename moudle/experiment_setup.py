import os
import pickle
import torch
import random
from hypothesis.BERTSUMEXT import ExtSummarizer
from hypothesis.BERTSUMEXT_FedSum import Fed_Sum_ExtSummarizer
from moudle.dataset import get_CNNDM_dataset, get_WikiHow_dataset, get_GovernmentReport_dataset, get_PubMed_dataset, \
    get_Mixtape_dataset,get_Reddit_dataset
from moudle.dataloader import get_FL_dataloader
from tool.logger import *


def Experiment_Create_dataset(param_dict, is_test=True):
    is_test = False if param_dict["tt"] == "0" else True
    logger.info(f"is test:{is_test}")
    dataset_name = [i.strip().lower() for i in param_dict['dataset_name'].split(",")]
    data_path = []
    get_dataset = []

    if "CNNDM".lower() in dataset_name:
        # data_path = "./dataset/CNNDM"
        # get_dataset = get_CNNDM_dataset
        data_path.append("./dataset/CNNDM")
        get_dataset.append(get_CNNDM_dataset)

    if "WikiHow".lower() in dataset_name:
        # data_path = "./dataset/WikiHow"
        # get_dataset = get_WikiHow_dataset
        data_path.append("./dataset/WikiHow")
        get_dataset.append(get_WikiHow_dataset)

    if ("GovernmentReport".lower() in dataset_name) or ("gr".lower() in dataset_name):
        # data_path = "./dataset/GovernmentReport"
        # get_dataset = get_GovernmentReport_dataset
        data_path.append("./dataset/GovernmentReport")
        get_dataset.append(get_GovernmentReport_dataset)

    if "PubMed".lower() in dataset_name:
        # data_path = "./dataset/PubMed"
        # get_dataset = get_PubMed_dataset
        data_path.append("./dataset/PubMed")
        get_dataset.append(get_PubMed_dataset)

    if "Reddit".lower() in dataset_name:
        # data_path = "./dataset/PubMed"
        # get_dataset = get_PubMed_dataset
        data_path.append("./dataset/Reddit")
        get_dataset.append(get_Reddit_dataset)

    if "Mixtape".lower() in dataset_name:
        all_data_paths = ["./dataset/PubMed", "./dataset/GovernmentReport", "./dataset/WikiHow", "./dataset/CNNDM"]
        data_paths = ["./dataset/GovernmentReport", "./dataset/WikiHow", "./dataset/PubMed", "./dataset/CNNDM"]
        logger.info(f"data_choose:{data_paths}")

    training_dataset = []
    validation_dataset = []
    testing_dataset = []

    if "Mixtape".lower() in dataset_name:
        get_dataset_func = get_Mixtape_dataset
        if is_test:
            training_dataset = get_dataset_func(data_paths, "train", param_dict, only_one=True)
            # validation_dataset = get_dataset_func(data_paths, "valid", param_dict, only_one=True)
            testing_dataset = get_dataset_func(data_paths, "test", param_dict, only_one=True)
        else:
            training_dataset = get_dataset_func(data_paths, "train", param_dict, only_one=False)
            # validation_dataset = get_dataset_func(data_paths, "valid", param_dict, only_one=False)
            testing_dataset = get_dataset_func(data_paths, "test", param_dict, only_one=False)
    else:
        for i in range(len(data_path)):

            d = data_path[i]
            get_dataset_func = get_dataset[i]
            if is_test:
                training_dataset.append(get_dataset_func(d, "train", param_dict, only_one=False))
                # validation_dataset.append(get_dataset_func(d, "valid", param_dict, only_one=True))
                testing_dataset += get_dataset_func(d, "test", param_dict, only_one=False)
            else:
                training_dataset.append(get_dataset_func(d, "train", param_dict, only_one=False))
                # validation_dataset.append(get_dataset_func(d, "valid", param_dict, only_one=False))
                testing_dataset += get_dataset_func(d, "test", param_dict, only_one=False)

    return training_dataset, validation_dataset, testing_dataset


def Experiment_Create_dataloader(param_dict, training_dataset, validation_dataset, testing_dataset,
                                 split_strategy="Uniform"):
    num_clients_K = param_dict['num_clients_K']
    batch_size = param_dict['batch_size']

    if param_dict['dataset_name'] == "Mixtape":
        training_dataloaders, client_dataset_list = get_FL_dataloader(param_dict,
                                                                      training_dataset, num_clients_K,
                                                                      split_strategy=split_strategy,
                                                                      do_train=True, batch_size=batch_size,
                                                                      num_workers=0, do_shuffle=True,
                                                                      corpus_type="train"
                                                                      )

        # validation_dataloaders, _ = get_FL_dataloader(param_dict,
        #     validation_dataset, num_clients_K, split_strategy=split_strategy,
        #     do_train=True, batch_size=batch_size, num_workers=0, do_shuffle=True,corpus_type="val"
        # )

        testing_dataloader = get_FL_dataloader(param_dict,
                                               testing_dataset, num_clients_K, split_strategy="Uniform",
                                               do_train=False, batch_size=batch_size, num_workers=0, corpus_type="test"
                                               )
    else:
        print(len(training_dataset))
        # 一个领域的数据被存储到list的一个项中
        data_field_number = len(training_dataset)
        logger.info(f"### Data_Field_Number: {data_field_number} ###")

        if data_field_number == 1:
            training_dataloaders, client_dataset_list = get_FL_dataloader(param_dict,
                                                                          training_dataset[-1], num_clients_K,
                                                                          split_strategy=split_strategy,
                                                                          do_train=True, batch_size=batch_size,
                                                                          num_workers=0, do_shuffle=True,
                                                                          corpus_type="train"
                                                                          )
            # validation_dataloaders, _ = get_FL_dataloader(param_dict,
            #     validation_dataset[-1], num_clients_K, split_strategy=split_strategy,
            #     do_train=True, batch_size=batch_size, num_workers=0, do_shuffle=True,corpus_type="val"
            # )

        else:
            training_dataloaders = []
            client_dataset_list = []
            validation_dataloaders = []

            filed_size = [num_clients_K // data_field_number for _ in range(data_field_number)]
            logger.info(f"filed_size:{filed_size}")
            filed_size[-1] += num_clients_K % data_field_number

            for i in range(data_field_number):
                td, cd = get_FL_dataloader(param_dict,
                                           training_dataset[i], filed_size[i], split_strategy=split_strategy,
                                           do_train=True, batch_size=batch_size, num_workers=2, do_shuffle=True
                                           )

                vd, _ = get_FL_dataloader(param_dict,
                                          validation_dataset[i], filed_size[i], split_strategy=split_strategy,
                                          do_train=True, batch_size=batch_size, num_workers=2, do_shuffle=True
                                          )

                training_dataloaders += td
                client_dataset_list += cd
                validation_dataloaders += vd
        testing_dataloader = get_FL_dataloader(param_dict,
                                               testing_dataset, num_clients_K, split_strategy="Uniform",
                                               do_train=False, batch_size=int(0.25*batch_size), num_workers=0, corpus_type="test"
                                               )

    # return training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader
    return training_dataloaders, client_dataset_list, testing_dataloader


def Experiment_Create_model(param_dict):
    if "BERTSUMEXT".lower() in param_dict['hypothesis'].lower():
        if "FedSum".lower() in param_dict['algorithm'].lower():
            logger.info("Model construction (FedSum_BERTSUMEXT)")
            model = Fed_Sum_ExtSummarizer(classifier_type=param_dict['classifier_type'])
        elif "FedProto".lower() in param_dict['algorithm'].lower():
            logger.info("Model construction (FedSum_BERTSUMEXT)")
            model = Fed_Sum_ExtSummarizer(classifier_type=param_dict['classifier_type'])
        else:
            logger.info("Model construction (BERTSUMEXT)")
            # model = ExtSummarizer(classifier_type=param_dict['classifier_type'], forzen=param_dict['forzen'])
            model = ExtSummarizer(classifier_type=param_dict['classifier_type'])
    else:
        logger.info("No Model Name! Construction Fail!")
        raise ValueError(f'''Wrong model name:{param_dict['hypothesis']} It should be in the following type:
                    [BERTSUMEXT | XXXX] ''')
    # model.to(param_dict['device'])
    return model


def Experiment_Reload_model(checkpoint_path):
    model = torch.load(checkpoint_path)
    return model
