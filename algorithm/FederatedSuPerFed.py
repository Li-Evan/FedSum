import random

import torch
import numpy as np
import copy

from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model
from algorithm.Optimizers import BERTSUMEXT_Optimizer

from algorithm.client_selection import client_selection

from collections import OrderedDict, defaultdict
from functools import partial

import torch.optim as optim


# Federated SuPerFed.json with BERTSUM
def Fed_SuPerFed_BERTSUMEXT(device,
                            global_model,
                            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                            training_dataloaders,
                            training_dataset,
                            client_dataset_list,
                            param_dict):
    # training_dataset_size = len(training_dataset)
    if (param_dict["dataset_name"] == "Mixtape"):
        training_dataset_size = len(training_dataset)
    else:
        training_dataset_size = sum(len(_) for _ in training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    # Parameter Initialization
    federated_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]  # 本地的联邦交互部分
    local_model_list = copy.deepcopy(federated_model_list)  # 本地的个性化部分（不交互）

    criterion = torch.nn.BCELoss(reduction='none')

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')

    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    # communication_round_I 是指总的通信轮次
    for iter_t in range(communication_round_I):
        # Simulate Client Parallel
        for i in range(num_clients_K):
            # get the mixed model
            local_model = local_model_list[i]
            federated_model = federated_model_list[i]

            local_model.train()
            federated_model.train()

            start_mix = True if iter_t > param_dict["L"] * algorithm_epoch_T else False
            if start_mix:
                lamda = random.random()
            else:
                lamda = 0
            mixed_model = copy.deepcopy(local_model)
            for param_mixed, param_l, param_f in zip(mixed_model.parameters(), local_model.parameters(),
                                                     federated_model.parameters()):
                param_mixed.data = (1 - lamda) * param_f.data + lamda * param_l.data
            mixed_model.train()
            mixed_model.zero_grad()
            mixed_model.to(device)

            # optimizer initialization
            optimizer = BERTSUMEXT_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)

            optimizer.set_parameters(list(mixed_model.named_parameters()))

            # local option
            client_i_dataloader = training_dataloaders[i]

            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_K};  ##########")
            for batch_index, batch in enumerate(client_i_dataloader):
                mixed_model.zero_grad()
                src = batch['src'].to(device)
                labels = batch['src_sent_labels'].to(device)
                segs = batch['segs'].to(device)
                clss = batch['clss'].to(device)
                mask = batch['mask_src'].to(device)
                mask_cls = batch['mask_cls'].to(device)

                _, _ = local_model(src, segs, clss, mask, mask_cls)
                __, __ = federated_model(src, segs, clss, mask, mask_cls)
                sent_scores, mask = mixed_model(src, segs, clss, mask, mask_cls)

                # part1 of the final loss : ordinary loss function
                loss = criterion(sent_scores, labels.float())  # 现在的loss是一个向量，表示对真实y和预测y的向量的对应位置求交叉熵
                loss = (loss * mask.float()).sum()

                # part2 of the final loss(calculate proximity regularization term toward global model) : mu||w^f_i - w^g||^2
                if param_dict["mu"] > 0:
                    for param_f, param_g in zip(federated_model.parameters(), global_model.parameters()):
                        loss += param_dict["mu"] * (torch.norm(param_f.data - param_g.data, p=2) ** 2)

                # part3 of the final loss(subspace construction) : v*cossim^2(w^f_i,w^l_i)
                if start_mix:
                    numerator, norm_1, norm_2 = 0, 0, 0
                    for param_f, param_l in zip(federated_model.parameters(), local_model.parameters()):
                        numerator += (param_f.data * param_l.data).add(1e-6).sum()
                        norm_1 += param_f.data.pow(2).sum()
                        norm_2 += param_l.data.pow(2).sum()
                    cos_sim = numerator.pow(2).div(norm_1 * norm_2)
                    loss += param_dict["nu"] * cos_sim

                logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                            f"Client: {i + 1} / {num_clients_K}; "
                            f"Batch: {batch_index}; Loss: {loss.data} ##########")
                (loss / loss.numel()).backward()  # 标准写法不用管
                optimizer.step()
                local_model_grad = torch.autograd.grad(loss, local_model.parameters())
                federated_model_grad = torch.autograd.grad(loss, federated_model.parameters())
                local_model = local_model - param_dict["lr"] * local_model_grad
                federated_model = federated_model - param_dict["lr"] * federated_model_grad
                # optimizer.step()  # 更新的是 mixed_model 的参数，需要手动把 local_model 和 federated_model 的参数更新

            # Upgrade the local model list
            local_model_list[i] = local_model
            federated_model_list[i] = federated_model

        # Communicate
        if (iter_t + 1) % communication_round_I == 0:
            logger.info(f"********** Communicate: {(iter_t + 1) / communication_round_I} **********")
            # Client selection
            logger.info(f"********** Client selection **********")
            idxs_users = client_selection(
                client_num=num_clients_K,
                fraction=FL_fraction,
                dataset_size=training_dataset_size,
                client_dataset_size_list=client_datasets_size_list,
                drop_rate=FL_drop_rate,
                style="FedAvg",
            )
            logger.info(f"********** Select client list: {idxs_users} **********")

            # Global operation
            logger.info("********** Parameter aggregation **********")

            pre_param = np.array(get_parameters(global_model), dtype=object)
            new_global_params = 0 * pre_param

            selected_training_dataset_size = sum(client_datasets_size_list[id] for id in idxs_users)
            # the weight of each selected client dependent on the numbers of dataset it has
            for id in idxs_users:
                selected_model = federated_model_list[id]
                ratio = client_datasets_size_list[id] / selected_training_dataset_size
                selected_param = get_parameters(selected_model)
                weighted_arr = np.array(selected_param, dtype=object) * ratio
                new_global_params += weighted_arr

            set_parameters(global_model, new_global_params)

            # Parameter Distribution
            logger.info("********** Parameter distribution **********")
            local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

        # Save model
        if (iter_t % param_dict["save_checkpoint_epochs"] == 0) and iter_t != 0:
            save_model(param_dict, global_model, local_model_list, iter_t)

    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list
