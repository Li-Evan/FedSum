import torch
import numpy as np
import copy
import gc
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model, Testing_ROUGE
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection
import math

# FedratedRep with BERTSUM
def Fed_Rep_BERTSUMEXT(device,
                       global_model,
                       algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                       training_dataloaders,
                       training_dataset,
                       client_dataset_list,
                       param_dict,
                       testing_dataloader):
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
            style="FedAvg",
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

            try:
                # 先更新分类头
                logger.info(f"########## Update the local classification head ##########")
                # 本地训练分类头之前先把特征提取器的梯度截断
                for param in model.bert.model.parameters():
                    param.requires_grad = False
                for param in model.ext_layer.parameters():
                    param.requires_grad = True

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
                        sub_batch_size = param_dict['batch_size']
                        average_one_sample_loss_in_batch = 0
                        sub_batch_loss_list = []
                        for i in range(0, len(src), sub_batch_size):
                            # logger.info(f'Max memory usage: {torch.cuda.memory_reserved() / 1024 ** 2} MB')
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
                        average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                            client_datasets_size_list[id] / param_dict['batch_size'])
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
                        # break

                    # 不要删，这行是表示先不回传小分批loss，最后再一整批loss回传, 搭配上面
                    # average_one_sample_loss_in_epoch = epoch_total_loss / client_datasets_size_list[id]

                    logger.info(f"### Communication Round: {iter_t + 1} / {communication_round_I}; "
                                f"Client: {id} / {num_clients_K}; "
                                f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_sample_loss_in_epoch}  ####")

                    # if epoch+1 == (algorithm_epoch_T/2):
                    #     Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, model,
                    #                   epoch+1, iter_t)
                    #     model.to(device)

                    # record_time = math.ceil(algorithm_epoch_T*(0.2 if algorithm_epoch_T > 200 else 0.4))
                    # if (epoch + 1) % record_time == 0:
                    #     logger.info(f"########## Rouge_Testing Epoch: {epoch+1}; "
                    #             f"Client: {id} / {num_clients_K}; ")
                    #     Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, model,
                    #                        param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])
            except RuntimeError:
                break

            try:
                # 再更新特征提取器
                logger.info(f"########## Update local base model ##########")
                # 本地训练之前先把分类头的梯度截断
                for param in model.bert.model.parameters():
                    param.requires_grad = True
                for param in model.ext_layer.parameters():
                    param.requires_grad = False

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
                        sub_batch_loss_list = []
                        for i in range(0, len(src), sub_batch_size):
                            # logger.info(f'Max memory usage: {torch.cuda.memory_reserved() / 1024 ** 2} MB')
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
                        average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / (
                                    client_datasets_size_list[id] / param_dict['batch_size'])

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
                        # break

                    # 不要删，这行是表示先不回传小分批loss，最后再一整批loss回传, 搭配上面
                    # average_one_sample_loss_in_epoch = epoch_total_loss / client_datasets_size_list[id]

                    logger.info(f"### Communication Round: {iter_t + 1} / {communication_round_I}; "
                                f"Client: {id} / {num_clients_K}; "
                                f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_sample_loss_in_epoch}  ####")
            except RuntimeError:
                break

            # Upgrade the local model list
            local_model_list[id] = model.cpu()
            del model

        # Communicate
        logger.info(f"********** Communicate: {(iter_t + 1)} **********")

        # Global operation
        logger.info("********** Parameter aggregation **********")
        rate = 1 / len(idxs_users)
        # Server仅仅聚合base层部分的参数
        for idx in idxs_users:
            selected_client = local_model_list[idx]
            for global_param, client_param in zip(global_model.bert.model.parameters(), selected_client.bert.model.parameters()):
                global_param.data += rate * client_param.data

        if (iter_t + 1) != param_dict['communication_round_I']:
            logger.info(f"Global model testing at Communication {(iter_t + 1)}")
            logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; ")
            Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model,
                          param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])

        # Save model
        # TODO:现在是若干个通信轮次之后统一保存一次global和一次client，有必要的话可以改成在客户端的迭代里面保存，但感觉这个问题不大
        # if (iter_t) % param_dict["save_checkpoint_rounds"] == 0 and iter_t != 0:
        #     save_model(param_dict, global_model, local_model_list, iter_t)

    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list

