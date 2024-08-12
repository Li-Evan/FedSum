import os
import json
import argparse
import copy
import gc
import numpy as np
import math
import torch.nn.functional as F

from tool.logger import *
from tool.utils import check_and_make_the_path, get_parameters, set_parameters, str2bool, Testing_ROUGE
from moudle.experiment_setup import Experiment_Create_dataset
from moudle.experiment_setup import Experiment_Create_dataloader, Experiment_Create_model
from algorithm.FederatedAverage import Fed_AVG_BERTSUMEXT
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection


import heapq

def find_indices_of_top_k_values(lst, k):
    # 使用heapq.nlargest获取前k个最大值及其索引
    # 使用zip将值和索引配对
    # 使用sorted基于索引排序，获取有序的索引列表
    return [index for index, _ in heapq.nlargest(k, enumerate(lst), key=lambda x: x[1])]

def find_min_indices(lst, k):
    return [index for index, _ in heapq.nsmallest(k, enumerate(lst), key=lambda x: x[1])]

def Argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-algorithm", default='FedAvg', type=str)
    parser.add_argument("-learning_rate", default=5e-3, type=float)  # follow邱锡鹏
    parser.add_argument("-optimize_method", default='sgd', type=str)
    parser.add_argument("-dataset", default='CNNDM', type=str)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-train_from", default='')
    # parser.add_argument("-tt", default="0", type=str, help="tt=1进行小批量运算，tt=0进行全量运算，详情在dataset.py")
    parser.add_argument("-tt", default="few", type=str, help="tt=1进行小批量运算，tt=0进行全量运算，详情在dataset.py")

    parser.add_argument("-batch_size", default=32, type=int, help="batch size")  # follow邱锡鹏
    # parser.add_argument("-cuda", default="0", type=str, help="cuda")
    parser.add_argument("-cuda", default="1", type=str, help="cuda")

    args = parser.parse_args()
    param_dict = vars(args)
    param_dict["CUDA_VISIBLE_DEVICES"] = param_dict["cuda"]
    return param_dict


def Fed_AVG_BERTSUMEXT(device,
                       global_model,
                       algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                       training_dataloaders,
                       training_dataset,
                       client_dataset_list,
                       param_dict,
                       testing_dataloader=None):
    import torch
    # training_dataset_size = len(training_dataset)
    if (param_dict["dataset_name"] == "Mixtape"):
        training_dataset_size = len(training_dataset)
    else:
        training_dataset_size = sum(len(_) for _ in training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    # Parameter Initialization
    local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]


    criterion = torch.nn.BCELoss(reduction='none')

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')

    selected_num = max(int(FL_fraction * num_clients_K), 1)
    Max_idxs_users = find_indices_of_top_k_values(client_datasets_size_list, selected_num)
    Min_idxs_users = find_min_indices(client_datasets_size_list, selected_num)
    Ordered_idxs_users = find_min_indices(client_datasets_size_list, selected_num * communication_round_I)

    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    # communication_round_I 是指总的通信轮次
    for iter_t in range(communication_round_I):
        # 先选客户端，只对选中的客戶下发模型
        # Client Selection
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedProx",
        )


        # 下发模型
        for id in idxs_users:
            local_model_list[id] = copy.deepcopy(global_model)

        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        # Simulate Client Parallel
        for id in idxs_users:
            # Local Initialization
            model = local_model_list[id]
            model.train()
            model.to(device)
            optimizer = BERTSUMEXT_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]


            # Local Training
            for epoch in range(algorithm_epoch_T):
                epoch_total_loss = 0
                average_one_sample_loss_in_epoch = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                for batch_index, batch in enumerate(client_i_dataloader):
                    src = batch['src'].to(device)
                    labels = batch['src_sent_labels'].to(device)
                    segs = batch['segs'].to(device)
                    clss = batch['clss'].to(device)
                    mask = batch['mask_src'].to(device)
                    mask_cls = batch['mask_cls'].to(device)
                    sub_batch_size = 8
                    average_one_sample_loss_in_batch = 0

                    for i in range(0, len(src), sub_batch_size):
                        sbatch_size = src[i:i + sub_batch_size].shape[0]  # 获取当前批次的样本数量
                        tmp_sent_scores, tmp_mask = model(src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))
                        # 注意，criterion函数并没有进行reduction操作，loss的尺寸：【sub_batch_size，sub_batch的大句子数目】
                        loss = criterion(tmp_sent_scores, labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())
                        # sub_batch_loss_list.append(loss)  #不要删，这行是表示先不回传小分批loss，最后再一整批loss回传，搭配下面

                        # 为了避免爆显存，先回传loss
                        loss = torch.sum(loss) / src.shape[0]
                        average_one_sample_loss_in_batch += loss

                        loss.backward()
                        # torch.cuda.empty_cache()

                        # 输出参数梯度
                        tmp = []
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                tmp.append(torch.clone(param.grad.data))
                                # print(f'{name}: {param.grad.data}')

                    average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(client_datasets_size_list[id] / param_dict['batch_size'])

                    # 不要删，这行是表示先不回传小分批loss，最后再一整批loss回传, 搭配上面
                    # # batch_total_loss的尺寸：【batch_size，sub_batch的大句子数目】
                    # # batch_total_loss = torch.cat(sub_batch_loss_list, 0)
                    # # 一整个batch内，平均一个样本的损失
                    # averaged_one_sample_loss = torch.sum(batch_total_loss) / param_dict['batch_size']
                    # epoch_total_loss += torch.sum(batch_total_loss)
                    # # logger.info(f"### Avg one sample's Loss in one batch: {averaged_one_sample_loss}")
                    # averaged_one_sample_loss.backward()



                    # FedAvg算法一个batch就做一次更新
                    optimizer.step()
                    model.zero_grad()

                    del tmp_sent_scores, tmp_mask, src, labels, segs, clss, mask, mask_cls
                    gc.collect()
                    # torch.cuda.empty_cache()


                # 不要删，这行是表示先不回传小分批loss，最后再一整批loss回传, 搭配上面
                # average_one_sample_loss_in_epoch = epoch_total_loss / client_datasets_size_list[id]


                logger.info(f"### Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_sample_loss_in_epoch} ; ####")


            # Upgrade the local model list
            local_model_list[id] = model.cpu()
            del model
            # torch.cuda.empty_cache()

        # Communicate
        logger.info(f"********** Communicate: {(iter_t + 1)} **********")

        # Global operation
        logger.info("********** Parameter aggregation **********")
        theta_list = []
        for id in idxs_users:
            selected_model = local_model_list[id]
            theta_list.append(get_parameters(selected_model))

        theta_list = np.array(theta_list, dtype=object)
        theta_avg = np.mean(theta_list, 0).tolist()
        set_parameters(global_model, theta_avg)


        logger.info(f"Global model testing at Communication {(iter_t + 1)}")
        logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; ")
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])



    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list


def Experiment_Federated_Average(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader):
    device = param_dict['device']
    import torch

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_AVG_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")


def Experiment(param_dict, training_dataset, validation_dataset, testing_dataset):
    import torch
    # Create dataloader
    logger.info("Creating dataloader")
    # training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(
    #     param_dict, training_dataset, validation_dataset, testing_dataset, param_dict['split_strategy'])
    training_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(
        param_dict, training_dataset, validation_dataset, testing_dataset, param_dict['split_strategy'])

    # Model Construction
    # 为了避免过多的随机性影响，尽量保证在同一个初始的模型开始训练
    if "linear" in param_dict['classifier_type']:
        global_init_model_path = r"./save_path/global_model_init.pt"
        global_model = Experiment_Create_model(param_dict)
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            global_model.load_state_dict(torch.load(global_init_model_path).state_dict())
    elif "baseline" in param_dict['classifier_type']:
        global_init_model_path = r"./save_path/toy_global_model_init.pt"
        global_model = Experiment_Create_model(param_dict)
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            global_model.load_state_dict(torch.load(global_init_model_path).state_dict())
    else:
        raise AssertionError

    # if os.path.exists(param_dict["current_model"]):
    #     global_model.load_state_dict(torch.load(param_dict["current_model"]).state_dict())
    # global_model.to(param_dict['device'])
    logger.info("-----------------------------------------------------------------------------")
    print(f'Algorithm Name: {param_dict["algorithm"]}')

    # Federated Average
    logger.info("~~~~~~ Algorithm: Federated Average ~~~~~~")
    Experiment_Federated_Average(
        param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
        testing_dataloader
    )


def main(dataset_name, algorithm, hypothesis, classifier_type, device, param_dict):
    ## Hyper-params
    # Common Hyper-params
    with open("./json/COMMON.json", "r") as f:
        temp_dict = json.load(f)
    param_dict.update(**temp_dict)
    # Dataset Hyper-params
    dataset_name_list = dataset_name.split(",")
    for dataset_name in dataset_name_list:
        dataset_name = dataset_name.strip()
        if os.path.exists(os.path.join("./json/dataset/", dataset_name + ".json")):
            with open(os.path.join("./json/dataset/", dataset_name + ".json"), "r") as f:
                temp_dict = json.load(f)
            param_dict.update(**temp_dict)
    # Algorithm Hyper-params
    if os.path.exists(os.path.join("./json/algorithm/", algorithm + ".json")):
        with open(os.path.join("./json/algorithm/", algorithm + ".json"), "r") as f:
            temp_dict = json.load(f)
        param_dict.update(**temp_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['CUDA_VISIBLE_DEVICES']

    import torch
    if "gpu" in device.lower():
        param_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment
    else:
        param_dict['device'] = "cpu"

    FL_drop_rate_list = [0]  # 设置掉线率
    epoch_T_communication_I_list = [(2, 5)]  # 本地走T个epoch后进行一次通信，共走T*I个epoch，每次聚合都做性能测试
    split_strategy_list = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Dirichlet8", "Uniform"]
    # split_strategy_list = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform",  "Dirichlet8"] #不建议用Dir01，基本上分不到数据
    fraction_list = [0.2]
    # fraction_list = [0.1, 0.2, 0.3, 0.4]
    num_clients_K_list = [20]  # 设置客户端数目

    param_dict['dataset_name'] = dataset_name
    param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis
    param_dict['classifier_type'] = classifier_type

    # Skipping the unnecessary loop
    # TODO 这里留给一些可以跳过部分循环的算法
    # if "算法名".lower() not in algorithm.lower():
    #     FL_drop_rate_list = [0]




    # Serial number of experiment
    Experiment_NO = 1
    # total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list)
    total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list) * len(
        fraction_list) * len(num_clients_K_list)

    # Create dataset
    logger.info("Creating dataset")
    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(param_dict)


    # Main Loop
    for split_strategy in split_strategy_list:
        for FL_drop_rate in FL_drop_rate_list:
            param_dict['FL_drop_rate'] = FL_drop_rate
            for algorithm_epoch_T, communication_round_I in epoch_T_communication_I_list:
                for fraction in fraction_list:
                    for num_clients_K in num_clients_K_list:

                        param_dict['split_strategy'] = split_strategy
                        param_dict['num_clients_K'] = num_clients_K
                        param_dict['algorithm_epoch_T'] = algorithm_epoch_T
                        param_dict['communication_round_I'] = communication_round_I
                        param_dict['FL_fraction'] = fraction

                        ################################################################################################
                        # Create the log
                        log_path = os.path.join("./log_path", "toy4MD", param_dict['dataset_name'], param_dict['split_strategy'],
                                                param_dict['algorithm'], param_dict['hypothesis'] + "(" + param_dict[
                                                    'classifier_type'] + "_Head)",
                                                str(num_clients_K) + "Clients")
                        check_and_make_the_path(log_path)
                        log_path = os.path.join(log_path, str(Experiment_NO) + ".txt")
                        param_dict['log_path'] = log_path
                        file_handler = logging.FileHandler(log_path)
                        file_handler.setFormatter(formatter)
                        logger.addHandler(file_handler)
                        ################################################################################################
                        # Create the model path
                        model_path = os.path.join("./save_path", param_dict['dataset_name'],
                                                  param_dict['split_strategy'],
                                                  param_dict['algorithm'], param_dict['hypothesis'] + "(" + param_dict[
                                                      'classifier_type'] + "_Head)",
                                                  str(num_clients_K) + "Clients")
                        check_and_make_the_path(model_path)
                        param_dict['model_path'] = model_path
                        for k in range(param_dict["num_clients_K"]):
                            _ = os.path.join(model_path, "client_" + str(k + 1))
                            check_and_make_the_path(_)
                        logger.info(f"FL_drop_rate_list = %s ;" % FL_drop_rate_list
                                    + "epoch_T_communication_I_list = %s ;\n" % epoch_T_communication_I_list
                                    + "split_strategy_list = %s ; " % split_strategy_list
                                    + "fraction_list = %s" % fraction_list)
                        logger.info(f"Experiment {Experiment_NO}/{total_Experiment_NO} setup finish")
                        param_dict['Experiment_NO'] = str(Experiment_NO)
                        ################################################################################################
                        # Parameter announcement
                        logger.info("Parameter announcement")
                        for para_key in list(param_dict.keys()):
                            if "_common" in para_key:
                                continue
                            logger.info(f"****** {para_key} : {param_dict[para_key]} ******")
                        logger.info("-----------------------------------------------------------------------------")
                        ################################################################################################
                        # Experiment
                        # if Experiment_NO != 1:
                        Experiment(param_dict, training_dataset, validation_dataset, testing_dataset)
                        Experiment_NO += 1
                        logger.removeHandler(file_handler)
                        logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                        logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")


if __name__ == '__main__':
    param_dict = Argparse()
    main(dataset_name=param_dict['dataset'],
         algorithm=param_dict['algorithm'],
         hypothesis="BERTSUMEXT",
         # classifier_type = "random_transformer",  # not pre-trained BERT and Transformer head
         classifier_type="linear",  # pre-trained BERT and Linear head
         # classifier_type="baseline",  # not pre-trained BERT and Linear head
         device="gpu",
         param_dict=param_dict)
