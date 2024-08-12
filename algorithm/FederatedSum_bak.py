import torch
import numpy as np
import copy
import gc
import math
from tqdm import tqdm, trange
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model, Testing_ROUGE
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


# 冲突客户检测
# 进行标准联邦训练后，客户上传本地更新梯度或近似梯度。
# 服务器根据上传的梯度之间的余弦值构建客户更新相似表。
# 服务器根据相似表值是否大于零，判断两两客户之间是否为冲突组。
# 根据上述判断结果构建冲突记录表。
# 知识回顾训练
# 将冲突组内的客户分类模组参数进行基于可训练参数α的线性组合，记为分类知识融合模组。
# 联邦服务器对模型的表征模块进行联邦聚合，得到最新表征模型。
# 根据冲突记录表，将最新表征模型与多个融合模组重新分发到冲突组的客户。
# 重新收到参数的客户冻结最新表征模型，并进行本地数据推理，得到本地特征表示。
# 将多个本地特征输入到融合模组中，训练α，得到最新融合模组。
# 冲突客户将融合模组重新上传，联邦服务器将冲突组的分类器模块用融合模组替换。
# 联邦服务器对模型的分类器模块进行联邦聚合，得到最新分类模型。

# Federated Summ with BERTSUM model
def Fed_Sum_BERTSUMEXT(device,
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



        global_hyper_knowledge_prototype_list = []
        global_label_0_prototype_list = []
        global_label_1_prototype_list = []
        global_predict_0_prototype_list = []
        global_predict_1_prototype_list = []

        global_feature_list = []  # 存放每个客户的 高级prototype，这个里面的内容是乘了数据量多项式分布权重的
        raw_global_feature_list = []  # 存放每个客户的 高级prototype，这里没有乘任何内容

        global_predict_0_feature_list = []
        global_predict_1_feature_list = []
        global_label_0_feature_list = []
        global_label_1_feature_list = []

        global_approximate_gradient_list = []

        # 下发模型
        for id in idxs_users:
            local_model_list[id] = copy.deepcopy(global_model)

        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        # Simulate Client Parallel
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            model = local_model_list[id]
            # backup_model = copy.deepcopy(model)

            model.train()
            model.to(device)
            optimizer = BERTSUMEXT_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            client_i_feature_list = []
            client_i_predict_0_feature_list = []
            client_i_predict_1_feature_list = []
            client_i_label_0_feature_list = []
            client_i_label_1_feature_list = []

            # Local Training
            for epoch in range(algorithm_epoch_T):
                average_one_sample_loss_in_epoch = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                for batch_index, batch in enumerate(client_i_dataloader):
                    feature_list = []
                    predict_0_feature_list = []
                    predict_1_feature_list = []
                    label_0_feature_list = []
                    label_1_feature_list = []
                    loss = 0

                    src = batch['src'].to(device)
                    labels = batch['src_sent_labels'].to(device)
                    segs = batch['segs'].to(device)
                    clss = batch['clss'].to(device)
                    mask = batch['mask_src'].to(device)
                    mask_cls = batch['mask_cls'].to(device)
                    sub_batch_size = 8
                    average_one_sample_loss_in_batch = 0
                    src_len = len(src)
                    total_sub_batch_number = int(math.ceil(src_len / sub_batch_size))
                    now_sub_batch_number = int(0)
                    for i in range(0, src_len, sub_batch_size):
                        now_sub_batch_number += 1
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

                        # 添加类原型素材
                        sent_label_flag = labels.gt(0.5)
                        for doc_index, doc in enumerate(tmp_sent_scores):
                            for sent_index, sent_flag in enumerate( doc.gt(0.5) ):
                                sent_feature = sents_vec[doc_index, sent_index]
                                if torch.all(sent_feature == 0):  #排除用于对齐输出长度的tensor
                                    continue
                                feature_list.append(sent_feature)
                                client_i_feature_list.append(sent_feature)

                                if sent_flag:
                                    # logger.info("Append predict 1 feature! ")
                                    predict_1_feature_list.append(sent_feature)
                                    client_i_predict_1_feature_list.append(sent_feature)
                                else:
                                    # logger.info("Append predict 0 feature! ")
                                    predict_0_feature_list.append(sent_feature)
                                    client_i_predict_0_feature_list.append(sent_feature)
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


                        # 先把除了最后一个sub batch之前的损失的梯度传回去，避免爆显存。
                        # 最后一个sub batch的损失的梯度先不传，保留它的计算图，后面方便用于修改loss值后的回传。
                        # if (i + sub_batch_size) < len(src):
                        if now_sub_batch_number != total_sub_batch_number:
                            average_one_sample_loss_in_batch += loss
                            loss.backward()


                    # 最后一个sub batch修改loss
                    (label_0_feature_gap, label_1_feature_gap, hyper_knowledge_gap,
                     predict_0_feature_gap, predict_1_feature_gap,
                     label_predict_0_feature_gap, label_predict_1_feature_gap,
                     local_predict_01_feature_gap, global_predict_01_feature_gap) = 0, 0, 0, 0, 0, 0, 0, 0, 0

                    with torch.no_grad():
                        # 全局-本地预测原型的差异
                        if len(predict_0_feature_list) != 0:
                            predict_0_prototype = torch.stack(predict_0_feature_list, dim=0).mean(dim=0)
                            if len(global_predict_0_prototype_list) != 0:
                                predict_0_feature_gap = torch.norm((global_predict_0_prototype_list[-1] - predict_0_prototype), p=2)
                                predict_0_feature_gap = predict_0_feature_gap ** 2
                        if len(predict_1_feature_list) != 0:
                            predict_1_prototype = torch.stack(predict_1_feature_list, dim=0).mean(dim=0)
                            if len(global_predict_1_prototype_list) != 0:
                                predict_1_feature_gap = torch.norm((global_predict_1_prototype_list[-1] - predict_1_prototype), p=2)
                                predict_1_feature_gap = predict_1_feature_gap ** 2

                        # 全局-本地类原型的差异
                        if len(label_0_feature_list) != 0:
                            label_0_prototype = torch.stack(label_0_feature_list, dim=0).mean(dim=0)
                            if len(global_label_0_prototype_list) != 0:
                                label_0_feature_gap = torch.norm((global_label_0_prototype_list[-1] - label_0_prototype), p=2)
                                label_0_feature_gap = label_0_feature_gap ** 2
                        if len(label_1_feature_list) != 0:
                            label_1_prototype = torch.stack(label_1_feature_list, dim=0).mean(dim=0)
                            if len(global_label_1_prototype_list) != 0:
                                label_1_feature_gap = torch.norm( (global_label_1_prototype_list[-1] - label_1_prototype), p=2)
                                label_1_feature_gap = label_1_feature_gap ** 2

                        # 本地预测原型与本地类原型的差异
                        if len(label_0_feature_list) != 0:
                            label_0_prototype = torch.stack(label_0_feature_list, dim=0).mean(dim=0)
                            if len(predict_0_feature_list) != 0:
                                predict_0_prototype = torch.stack(predict_0_feature_list, dim=0).mean(dim=0)
                                label_predict_0_feature_gap = torch.norm((label_0_prototype - predict_0_prototype), p=2) ** 2
                        if len(label_1_feature_list) != 0:
                            label_1_prototype = torch.stack(label_1_feature_list, dim=0).mean(dim=0)
                            if len(predict_1_feature_list) != 0:
                                predict_1_prototype = torch.stack(predict_1_feature_list, dim=0).mean(dim=0)
                                label_predict_1_feature_gap = torch.norm((label_1_prototype - predict_1_prototype), p=2) ** 2

                        # 希望提高预测的差异常度，拉大预测原型之间差距
                        if (len(predict_1_feature_list) != 0) and (len(predict_0_feature_list) != 0):
                            try:
                                local_predict_01_feature_gap = torch.norm((predict_1_prototype - predict_0_prototype), p=2)
                                local_predict_01_feature_gap = local_predict_01_feature_gap ** 2
                            except Exception:
                                logger.info("local_predict_01_feature_gap error, set it as 0")
                                local_predict_01_feature_gap = 0
                            if (len(global_predict_0_prototype_list) != 0) and (len(global_predict_1_prototype_list) != 0):
                                try:
                                    global_predict_01_feature_gap = (torch.norm((predict_0_prototype - global_predict_1_prototype_list[-1]), p=2)
                                                           + torch.norm((predict_1_prototype - global_predict_0_prototype_list[-1]), p=2))
                                    global_predict_01_feature_gap = global_predict_01_feature_gap ** 2
                                except Exception:
                                    global_predict_01_feature_gap = 0
                                    logger.info("global_predict_01_feature_gap error, set it  as 0")

                        # 超原型差异
                        hyper_knowledge_prototype = torch.stack(feature_list, dim=0).mean(dim=0)
                        if len(global_hyper_knowledge_prototype_list) != 0:
                            hyper_knowledge_gap = torch.norm((global_hyper_knowledge_prototype_list[-1] - hyper_knowledge_prototype), p=2)
                            hyper_knowledge_gap = hyper_knowledge_gap ** 2

                    # lamda_list = [1, 1, 1, 1, 1, 1, 1, -1, -1]
                    lamda_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                    gap_list = [label_0_feature_gap, label_1_feature_gap, predict_0_feature_gap, predict_1_feature_gap,
                                label_predict_0_feature_gap, label_predict_1_feature_gap, hyper_knowledge_gap,
                                local_predict_01_feature_gap, global_predict_01_feature_gap
                                ]
                    # logger.info(f"### len(predict_0_feature_list) : {len(predict_0_feature_list)} ; "
                    #             f"len(predict_1_feature_list) : {len(predict_1_feature_list)}; "
                    #             f"len(label_0_feature_list) : {len(label_0_feature_list)}; "
                    #             f"len(label_1_feature_list) : {len(label_1_feature_list)}; ####")

                    # logger.info(f"### label_1_feature_gap : {label_1_feature_gap} ; "
                    #             f"label_0_feature_gap : {label_0_feature_gap}; "
                    #             f"predict_1_feature_gap : {predict_1_feature_gap}; "
                    #             f"predict_0_feature_gap : {predict_0_feature_gap}; "
                    #             f"label_predict_1_feature_gap : {label_predict_1_feature_gap}; "
                    #             f"label_predict_0_feature_gap : {label_predict_0_feature_gap}; "
                    #             f"hyper_knowledge_gap : {hyper_knowledge_gap}; "
                    #             f"local_predict_01_feature_gap : {local_predict_01_feature_gap}; "
                    #             f"global_predict_01_feature_gap : {global_predict_01_feature_gap} ####")
                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * gap_list[index]

                    average_one_sample_loss_in_batch += loss
                    average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / ( client_datasets_size_list[id] / param_dict['batch_size'])

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

            client_i_hyper_knowledge_prototype = torch.stack(client_i_feature_list, dim=0).mean(dim=0)
            # 先做加权，内层循环容易获得权重，方便后续操作
            global_feature_list.append(client_i_aggregation_weight * client_i_hyper_knowledge_prototype)
            raw_global_feature_list.append(client_i_hyper_knowledge_prototype)

            if len(client_i_predict_0_feature_list) != 0:
                client_i_predict_0_prototype = torch.stack(client_i_predict_0_feature_list, dim=0).mean(dim=0)
                # 先做加权，内层循环容易获得权重，方便后续操作
                global_predict_0_feature_list.append(client_i_aggregation_weight * client_i_predict_0_prototype)
            if len(client_i_predict_1_feature_list) != 0:
                client_i_predict_1_prototype = torch.stack(client_i_predict_1_feature_list, dim=0).mean(dim=0)
                # 先做加权，内层循环容易获得权重，方便后续操作
                global_predict_1_feature_list.append(client_i_aggregation_weight * client_i_predict_1_prototype)

            if len(client_i_label_0_feature_list) != 0:
                client_i_label_0_prototype = torch.stack(client_i_label_0_feature_list, dim=0).mean(dim=0)
                # 先做加权，内层循环容易获得权重，方便后续操作
                global_label_0_feature_list.append(client_i_aggregation_weight * client_i_label_0_prototype)

            if len(client_i_label_1_feature_list) != 0:
                client_i_label_1_prototype = torch.stack(client_i_label_1_feature_list, dim=0).mean(dim=0)
                # 先做加权，内层循环容易获得权重，方便后续操作
                global_label_1_feature_list.append(client_i_aggregation_weight * client_i_label_1_prototype)

            # Upgrade the local model list
            local_model_list[id] = model.cpu()




            # with torch.no_grad():
            #     new_param = torch.nn.utils.parameters_to_vector(model)
            #     old_param = torch.nn.utils.parameters_to_vector(backup_model)
            #     approximate_gradient = new_param - old_param
            #     global_approximate_gradient_list.append(approximate_gradient)
            # del model
            # del backup_model


            # torch.cuda.empty_cache()

        # Communicate
        logger.info(f"********** Communicate: {(iter_t + 1)} **********")

        # Global operation

        logger.info("********** Prototype aggregation **********")
        (global_hyper_knowledge_prototype, global_predict_0_prototype, global_predict_1_prototype,
         global_label_0_prototype,  global_label_1_prototype) = 0, 0, 0, 0, 0
        global_hyper_knowledge_prototype_list = []
        global_predict_0_prototype_list = []
        global_predict_1_prototype_list = []

        # 前面已经乘过权重了，所以这里只需要加起来即可
        for index, proto in enumerate(global_feature_list):
            global_hyper_knowledge_prototype += proto
        global_hyper_knowledge_prototype_list.append(global_hyper_knowledge_prototype)  # 更新全局的各种原型
        if len(global_predict_0_feature_list) != 0:
            for proto in global_predict_0_feature_list:
                global_predict_0_prototype += proto
            global_predict_0_prototype_list.append(global_predict_0_prototype)  # 更新全局的各种原型
        if len(global_predict_1_feature_list) != 0:
            for proto in global_predict_1_feature_list:
                global_predict_1_prototype += proto
            global_predict_1_prototype_list.append(global_predict_1_prototype)  # 更新全局的各种原型
        if len(global_label_0_feature_list) != 0:
            for proto in global_label_0_feature_list:
                global_label_0_prototype += proto
            global_label_0_prototype_list.append(global_label_0_prototype)  # 更新全局的各种原型
        if len(global_label_1_feature_list) != 0:
            for proto in global_label_1_feature_list:
                global_label_1_prototype += proto
            global_label_1_prototype_list.append(global_label_1_prototype)  # 更新全局的各种原型

        logger.info("********** Parameter aggregation **********")
        theta_list = []
        for id in idxs_users:
            selected_model = local_model_list[id]
            theta_list.append(get_parameters(selected_model))

        theta_list = np.array(theta_list, dtype=object)
        # FedAvg新版论文的聚合权重是数据占比
        theta_avg = np.average(theta_list, axis=0, weights=[average_weight[j] for j in idxs_users]).tolist()
        # FedAvg旧版论文的聚合权重是平均
        # theta_avg = np.mean(theta_list, 0).tolist()
        set_parameters(global_model, theta_avg)

        # # 冲突检测
        # conflict_pair_list = []
        # global_approximate_matrix = torch.cat(global_approximate_gradient_list, 0)  # 尺寸:【被选中联邦的客户数目， 一个模型的参数量】
        # weight_simi_matrix = torch.cosine_similarity(global_approximate_matrix.unsqueeze(1),
        #                                       global_approximate_matrix.unsqueeze(0), dim=-1)
        #
        # weight_simi_matrix = torch.cosine_similarity(global_approximate_matrix.unsqueeze(1),
        #                                              global_approximate_matrix.unsqueeze(0), dim=-1)
        #
        # for m in range(len(client_selection)):
        #     for n in range(1, len(client_selection)):
        #         sim = weight_simi_matrix[m][n]
        #         print(f"client {idxs_users[m]} and client {idxs_users[n]} sim is : {sim}")
        #         if sim < 0:
        #             print(f"client {idxs_users[m]} and client {idxs_users[n]} is conflict")
        #             conflict_pair_list.append((m, n))

        # 通信后检测性能
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

