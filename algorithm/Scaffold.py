import torch
import numpy as np
import copy
import gc
import math
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model, Testing_ROUGE
from algorithm.client_selection import client_selection
from algorithm.Optimizers import Scaffold_Optimizer


# Federated Average with BERTSUM
def Scaffold_BERTSUMEXT(device,
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
        training_dataset_size = sum(len(i) for i in training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Parameter Initialization
    # 将Scaffold论文提出的所有控制变量都初始化为0
    for k, v in global_model.named_parameters():
        global_model.control[k] = torch.zeros_like(v.data)
        global_model.delta_control[k] = torch.zeros_like(v.data)
        global_model.delta_y[k] = torch.zeros_like(v.data)

    logger.info("Model copy! Be care of your Memory!")
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
        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        # 下发模型
        for id in idxs_users:
            local_model_list[id] = copy.deepcopy(global_model)


        # Simulate Client Parallel
        for id in idxs_users:
            model = local_model_list[id]
            model.train()
            model.to(device)

            optimizer = Scaffold_Optimizer(model.parameters(), method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            # 提前保存一份没经过训练的全局模型的参数
            x = copy.deepcopy(model)

            # Local Training
            # for epoch in tqdm(range(algorithm_epoch_T), desc="Epoch"):
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
                    sub_batch_size = 16
                    average_one_sample_loss_in_batch = 0
                    sub_batch_loss_list = []
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
                        average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / (
                                client_datasets_size_list[id] / param_dict['batch_size'])

                    # 不要删，这行是表示先不回传小分批loss，最后再一整批loss回传, 搭配上面
                    # # batch_total_loss的尺寸：【batch_size，sub_batch的大句子数目】
                    # batch_total_loss = torch.cat(sub_batch_loss_list, 0)
                    # # 一整个batch内，平均一个样本的损失
                    # averaged_one_sample_loss = torch.sum(batch_total_loss) / param_dict['batch_size']
                    # epoch_total_loss += torch.sum(batch_total_loss)
                    # # logger.info(f"### Avg one sample's Loss in one batch: {averaged_one_sample_loss}")
                    # averaged_one_sample_loss.backward()

                    # FedAvg算法一个batch就做一次更新
                    optimizer.step(device, global_model.control, model.control)
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

            # 更新参数ci
            # temp保存了客户端模型参数y_i
            temp = {}
            for k, v in model.named_parameters():
                temp[k] = v.data

            # TODO:temp[k] 就是y_i, v.data就是x（对应于论文中公式的符号）
            for k, v in x.named_parameters():
                # print(model.control[k].is_cuda, global_model.control[k].is_cuda)
                model.control[k] = model.control[k].to(device)
                global_model.control[k] = global_model.control[k].to(device)
                x.control[k] = x.control[k].to(device)

                model.control[k] = model.control[k] - global_model.control[k] + (v.data - temp[k]) / (
                            algorithm_epoch_T * 0.005)
                model.delta_y[k] = temp[k] - v.data
                model.delta_control[k] = model.control[k] - x.control[k]

                model.control[k] = model.control[k].cpu()
                global_model.control[k] = global_model.control[k].cpu()
                x.control[k] = x.control[k].cpu()

            # Upgrade the local model list
            local_model_list[id] = model.cpu()
            del temp, x, model
            gc.collect()
            # torch.cuda.empty_cache()

        # Communicate
        logger.info(f"********** Communicate: {(iter_t + 1)} **********")

        # Global operation
        logger.info("********** Parameter aggregation **********")
        x = {}
        c = {}
        # init
        for k, v in local_model_list[0].named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)

        for j in idxs_users:
            for k, v in local_model_list[j].named_parameters():
                local_model_list[j].delta_y[k] = local_model_list[j].delta_y[k].to(device)
                local_model_list[j].delta_control[k] = local_model_list[j].delta_control[k].to(device)
                x[k] = x[k].to(device)
                c[k] = c[k].to(device)

                x[k] += local_model_list[j].delta_y[k] / len(idxs_users)
                c[k] += local_model_list[j].delta_control[k] / len(idxs_users)

                local_model_list[j].delta_y[k] = local_model_list[j].delta_y[k].cpu()
                local_model_list[j].delta_control[k] = local_model_list[j].delta_control[k].cpu()
                x[k] = x[k].cpu()
                c[k] = c[k].cpu()

        for k, v in global_model.named_parameters():
            v.data += x[k].data * param_dict["slr"]
            global_model.control[k].data += c[k].data

        if (iter_t + 1) != param_dict['communication_round_I']:

            logger.info(f"Global model testing:Communicate {(iter_t + 1)}")
            logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; ")
            Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])

        # Save model
        # TODO:现在是若干个通信轮次之后统一保存一次global和一次client，有必要的话可以改成在客户端的迭代里面保存，但感觉这个问题不大
        if (iter_t) % param_dict["save_checkpoint_rounds"] == 0 and iter_t != 0:
            save_model(param_dict, global_model, local_model_list, iter_t)
    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list
