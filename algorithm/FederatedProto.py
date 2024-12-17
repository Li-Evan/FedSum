import torch
import numpy as np
import copy
import gc
import math
from tqdm import tqdm, trange
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model, Testing_ROUGE_Personalize
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection

# Federated Proto with BERTSUM model
def Fed_PROTO_BERTSUMEXT(device,
                       global_model,
                       algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                       training_dataloaders,
                       training_dataset,
                       client_dataset_list,
                       param_dict,
                       testing_dataloader=None):
    # training_dataset_size = len(training_dataset)
    if (param_dict["dataset_name"] == "Mixtape"):
        training_dataset_size = len(training_dataset)
    else:
        training_dataset_size = sum(len(_) for _ in training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    average_weight = np.array([float(i / training_dataset_size) for i in client_datasets_size_list])

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

        global_label_0_prototype_list = []
        global_label_1_prototype_list = []

        global_label_0_feature_list = []
        global_label_1_feature_list = []


        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        # Simulate Client Parallel
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            model = local_model_list[id]

            model.train()
            model.to(device)
            optimizer = BERTSUMEXT_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            client_i_label_0_feature_list = []
            client_i_label_1_feature_list = []

            # Local Training
            for epoch in range(algorithm_epoch_T):
                average_one_sample_loss_in_epoch = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                for batch_index, batch in enumerate(client_i_dataloader):
                    label_0_feature_list = []
                    label_1_feature_list = []

                    src = batch['src'].to(device)
                    labels = batch['src_sent_labels'].to(device)
                    segs = batch['segs'].to(device)
                    clss = batch['clss'].to(device)
                    mask = batch['mask_src'].to(device)
                    mask_cls = batch['mask_cls'].to(device)
                    sub_batch_size = 8
                    average_one_sample_loss_in_batch = 0
                    src_len = len(src)
                    for i in range(0, src_len, sub_batch_size):
                        sbatch_size = src[i:i + sub_batch_size].shape[0]  # 获取当前批次的样本数量
                        # top_vec的尺寸：【sub_batch_size，最大输入字符数，768】
                        # sents_vec的尺寸：【sub_batch_size，批内最长句子数目，768】
                        top_vec, sents_vec = model.only_PLM_forward(src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))

                        # tmp_sent_scores的尺寸：【sub_batch_size，批内最长句子数目】
                        tmp_sent_scores, tmp_mask = model.only_clf_forward(sents_vec, mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))

                        # 添加原型素材
                        sent_label_flag = labels.gt(0.5)
                        for doc_index, doc in enumerate(tmp_sent_scores):
                            for sent_index, sent_flag in enumerate( doc.gt(0.5) ):
                                sent_feature = sents_vec[doc_index, sent_index]
                                if torch.all(sent_feature == 0):  #排除用于对齐输出长度的tensor
                                    continue

                                if sent_label_flag[i:i + sbatch_size][doc_index][sent_index]:
                                    # logger.info("Append label 1 feature! ")
                                    label_1_feature_list.append(sent_feature)
                                    client_i_label_1_feature_list.append(sent_feature)
                                else:
                                    # logger.info("Append label 0 feature! ")
                                    label_0_feature_list.append(sent_feature)
                                    client_i_label_0_feature_list.append(sent_feature)

                        # 注意，criterion函数并没有进行reduction操作，sub_batch_loss的尺寸：【sub_batch_size，sub_batch内最长句子数目】
                        sub_batch_loss = criterion(tmp_sent_scores, labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())
                        loss = torch.sum(sub_batch_loss) / src.shape[0]

                        # 为了避免爆显存，先回传loss
                        average_one_sample_loss_in_batch += loss
                        loss.backward()



                    # 重新产生一次计算图，方便添加原型损失
                    top_vec, sents_vec = model.only_PLM_forward(src[0].reshape(1, -1),
                                                      segs[0].reshape(1, -1),
                                                      clss[0].reshape(1, -1),
                                                      mask[0].reshape(1, -1),
                                                      mask_cls[0].reshape(1, -1))
                    tmp_sent_scores, tmp_mask = model.only_clf_forward(sents_vec, mask_cls[0].reshape(1, -1))
                    sub_batch_loss = criterion(tmp_sent_scores, labels[0].reshape(1, -1).float())
                    loss = torch.sum(sub_batch_loss) * 0

                    # 计算gap
                    (label_0_feature_gap, label_1_feature_gap) = 0, 0

                    with torch.no_grad():
                         # 全局-本地类原型的差异
                        if len(label_0_feature_list) != 0:
                            label_0_prototype = torch.stack(label_0_feature_list, dim=0).mean(dim=0)
                            if len(global_label_0_prototype_list) != 0:
                                label_0_feature_gap = torch.norm((global_label_0_prototype_list[-1] - label_0_prototype),
                                                                 p=2)
                                label_0_feature_gap = label_0_feature_gap
                        if len(label_1_feature_list) != 0:
                            label_1_prototype = torch.stack(label_1_feature_list, dim=0).mean(dim=0)
                            if len(global_label_1_prototype_list) != 0:
                                label_1_feature_gap = torch.norm((global_label_1_prototype_list[-1] - label_1_prototype),
                                                                 p=2)
                                label_1_feature_gap = label_1_feature_gap

                    lamda_list = [1, 1]  # FedPro思路


                    gap_list = [label_0_feature_gap, label_1_feature_gap]

                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * gap_list[index]
                        average_one_sample_loss_in_batch += loss / src.shape[0]

                    average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                        client_datasets_size_list[id] / param_dict['batch_size'])
                    # 梯度回传
                    loss.backward()

                    # FedAvg算法一个batch就做一次更新
                    optimizer.step()
                    model.zero_grad()

                    del tmp_sent_scores, tmp_mask, src, labels, segs, clss, mask, mask_cls
                    gc.collect()
                    # torch.cuda.empty_cache()

                logger.info(f"### Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_sample_loss_in_epoch}  ####")


            # 计算客户的 类原型
            if len(client_i_label_0_feature_list) != 0:
                client_i_label_0_prototype = torch.stack(client_i_label_0_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_label_0_feature_list.append(client_i_aggregation_weight * client_i_label_0_prototype)

            if len(client_i_label_1_feature_list) != 0:
                client_i_label_1_prototype = torch.stack(client_i_label_1_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_label_1_feature_list.append(client_i_aggregation_weight * client_i_label_1_prototype)

            # Upgrade the local model list
            local_model_list[id] = model.cpu()

            del model
            # del backup_model

        # Communicate
        logger.info(f"********** Communicate: {(iter_t + 1)} **********")

        # Global operation
        logger.info("********** Prototype aggregation **********")
        (global_label_0_prototype,  global_label_1_prototype) = 0, 0

        # 前面已经乘过权重了，所以这里只需要加起来即可
        if len(global_label_0_feature_list) != 0:
            for proto in global_label_0_feature_list:
                global_label_0_prototype += proto
            global_label_0_prototype_list.append(global_label_0_prototype)  # 更新全局的各种原型
        if len(global_label_1_feature_list) != 0:
            for proto in global_label_1_feature_list:
                global_label_1_prototype += proto
            global_label_1_prototype_list.append(global_label_1_prototype)  # 更新全局的各种原型


        # 通信后检测性能
        if (iter_t + 1) != param_dict['communication_round_I']:
            logger.info(f"Global model testing at Communication {(iter_t + 1)}")
            logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; ")
            Testing_ROUGE_Personalize(param_dict, param_dict['device'], testing_dataloader, local_model_list, param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


        # Save model
        # TODO:现在是若干个通信轮次之后统一保存一次global和一次client，有必要的话可以改成在客户端的迭代里面保存，但感觉这个问题不大
        # if (iter_t) % param_dict["save_checkpoint_rounds"] == 0 and iter_t != 0:
        #     save_model(param_dict, global_model, local_model_list, iter_t)


    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list

