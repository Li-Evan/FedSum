import os.path

import torch
import numpy as np
import copy
import math
import gc
from moudle.experiment_setup import Experiment_Create_model
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model, test_rank, attenuation_function, Testing_ROUGE
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


def Fed_Giant(device,
              global_model,
              algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
              training_dataloaders,
              training_dataset,
              client_dataset_list,
              param_dict,
              testing_dataloader,
              reselect=1,
              pretrain_k=5):
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
    local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]

    criterion = torch.nn.BCELoss(reduction='none')
    final_rank = [1 / num_clients_K for _ in range(num_clients_K)]
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    # communication_round_I 是指总的通信轮次
    for iter_t in range(communication_round_I):
        # TODO reselect 表明经过多少轮就要重新计算选client的概率
        if (iter_t % reselect == 0):
            # 下发模型
            for id in range(num_clients_K):
                local_model_list[id] = copy.deepcopy(global_model).cpu()
            # 每个人再训练若干轮
            for id in range(num_clients_K):
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

                # TODO：pretrain_k衡量再次选择巨人之前要进行多少次局部训练
                for epoch in range(pretrain_k):
                    if epoch % 50 == 0:
                        print(epoch)
                    epoch_loss = 0
                    for batch_index, batch in enumerate(client_i_dataloader):  # 只需要直接执行完一个batch后break就行
                        model.train()
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
                            loss = criterion(tmp_sent_scores,
                                             labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())
                            loss = (loss * tmp_mask.float()).sum()
                            loss = loss / sbatch_size
                            loss_list.append((loss / loss.numel()).data)
                            (loss / loss.numel()).backward()

                        optimizer.step()
                        epoch_loss = sum(loss_list)/len(loss_list)
                        del tmp_sent_scores, tmp_mask, src, labels, segs, clss, mask, mask_cls
                        gc.collect()
                        torch.cuda.empty_cache()
                        break

                # Upgrade the local model list
                local_model_list[id] = copy.deepcopy(model).cpu()
            # TODO:计算每个 client 的 rouge 和 info entropy
            rouge_rank, shortcut_rank,redundant_rank = test_rank(param_dict, device, testing_dataloader, local_model_list, iter_t)
            # TODO:计算最终的排名并softmax
            final_rank = []
            for i in range(len(rouge_rank)):
                value = rouge_rank[i]+shortcut_rank[i]+redundant_rank[i]
                final_rank.append(value)
            # final_rank = [attenuation_function(shortcut_rank[i]) + rouge_rank[i] for i in range(len(shortcut_rank))]
            final_rank = torch.nn.functional.softmax(torch.tensor(final_rank), dim=0)

            # final_rank = torch.nn.functional.softmax(torch.tensor(rouge_rank), dim=0)
            # 计算概率列表的总和
            # total_probability = sum(final_rank)
            # print(total_probability)
            # # 使用总和来归一化概率列表的每个元素
            # final_rank = [prob / total_probability for prob in final_rank]
            logger.info(f"iter_t:{iter_t}，final_rank:{final_rank}")
        # client selected (由于softmax之后的值之和并不严格为1，所以不能用numpy提供的choice，自己写了一个)
        # idxs_users = _client_selection(final_rank, num_clients_K*FL_fraction)
        # print(idxs_users)
        # print(sum(final_rank))
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            probabilities=final_rank,
            style="giant",
        )

        logger.info(f"********** Communication Round: {iter_t + 1} **********")
        logger.info(f"********** Select client list: {idxs_users} **********")

        # send model
        # for id in idxs_users:
        #     local_model_list[id] = copy.deepcopy(global_model).cpu()

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
                    model.train()
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
                    epoch_loss = sum(loss_list)/len(loss_list)
                    del tmp_sent_scores, tmp_mask, src, labels, segs, clss, mask, mask_cls
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                logger.info(f"########## Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Loss: {epoch_loss.data} #######")

                record_time = math.ceil(algorithm_epoch_T * (0.2 if algorithm_epoch_T > 200 else 0.4))
                if (epoch + 1) % record_time == 0:
                    logger.info(f"########## Rouge_Testing epoch: {epoch + 1}; "
                                f"Client: {id} / {num_clients_K}; ")
                    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, model,
                                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])

            logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; "
                        f"Client: {id} / {num_clients_K}; ")
            Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, model,
                          param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])
            # Upgrade the local model list
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
        logger.info(f"Global model testing:Communicate {(iter_t + 1)}")
        logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; ")
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])

        # Save model
        # TODO:现在是若干个通信轮次之后统一保存一次global和一次client，有必要的话可以改成在客户端的迭代里面保存，但感觉这个问题不大
        # if (iter_t) % param_dict["save_checkpoint_rounds"] == 0 and iter_t != 0:
        #     save_model(param_dict, global_model, local_model_list, iter_t)
    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list
