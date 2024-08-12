import os.path

import torch
import numpy as np
import copy
import math
import gc

from moudle.experiment_setup import Experiment_Create_model
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

# Federated test with BERTSUM
def Fed_TEST(device,
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
        training_dataset_size = sum(len(i) for i in training_dataset)

    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]
    # TODO: 临时把前四个client的模型都换成预训练好的（强制生成巨人）
    print(os.path.join(os.getcwd(),"algorithm","model.pkl"))
    pretrain_model = torch.load(os.path.join(os.getcwd(),"algorithm","model.pkl"))
    # pretrain_model = torch.load(os.path.join(param_dict['model_path'] , "model.pkl") )
    for i in range(4):
        local_model_list[i] = copy.deepcopy(pretrain_model)

    criterion = torch.nn.BCELoss(reduction='none')

    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    # communication_round_I 是指总的通信轮次
    for iter_t in range(communication_round_I):
        # TODO:对强制生成的巨人生成较高的选择概率
        # client Client
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )

        logger.info(f"********** Communication Round: {iter_t + 1} **********")
        logger.info(f"********** Select client list: {idxs_users} **********")

        # send model
        for id in idxs_users:
            local_model_list[id] = copy.deepcopy(global_model).cpu()

        # Simulate Client Parallel
        for id in idxs_users:
            model = local_model_list[id]
            model.train()
            model.zero_grad()
            model.to(device)
            # logger.info(f'Max memory usage: {torch.cuda.memory_reserved() / 1024 ** 2} MB')
            optimizer = BERTSUMEXT_Optimizer(
                method="sgd", learning_rate=0.0001, max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))

            # local option
            client_i_dataloader = training_dataloaders[id]

            # TODO 现在一个epoch只需要跑一个batch
            for epoch in range(algorithm_epoch_T):
                epoch_loss = 0
                for batch_index, batch in enumerate(client_i_dataloader):  # 只需要直接执行完一个batch后break就行
                    model.zero_grad()
                    src = batch['src'].to(device)
                    labels = batch['src_sent_labels'].to(device)
                    segs = batch['segs'].to(device)
                    clss = batch['clss'].to(device)
                    mask = batch['mask_src'].to(device)
                    mask_cls = batch['mask_cls'].to(device)
                    loss_list = []
                    for i in range(0, len(src), 4):
                        sbatch_size = src[i:i + 4].shape[0]  # 获取当前批次的样本数量
                        tmp_sent_scores, tmp_mask = model(src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))
                        loss = criterion(tmp_sent_scores, labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())
                        loss = (loss * tmp_mask.float()).sum()
                        loss = loss / sbatch_size
                        loss_list.append((loss / loss.numel()).data)
                        (loss / loss.numel()).backward()

                    optimizer.step()
                    epoch_loss = sum(loss_list)
                    del tmp_sent_scores, tmp_mask, src, labels, segs, clss, mask, mask_cls
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                logger.info(f"##########SeparateTraining"
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Loss: {epoch_loss.data} ##########")
                # record_time = math.ceil(algorithm_epoch_T / 4)  # 根据训练步数决定多久记一次loss 一个communicattion记录4次
                # if (epoch + 1) % record_time == 0:
                #     logger.info(f"########## Communication Round: {iter_t + 1} / {communication_round_I}; "
                #                 f"Client: {id} / {num_clients_K}; "
                #                 f"Epoch: {epoch + 1}; Loss: {epoch_loss.data} ##########")

            # Upgrade the local model list
            local_model_list[id] = copy.deepcopy(model).cpu()

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

        # Save model
        # TODO:现在是若干个通信轮次之后统一保存一次global和一次client，有必要的话可以改成在客户端的迭代里面保存，但感觉这个问题不大
        # if (iter_t) % param_dict["save_checkpoint_rounds"] == 0 and iter_t != 0:
        #     save_model(param_dict, global_model, local_model_list, iter_t)
    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list
