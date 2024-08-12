import torch
import numpy as np
import copy
import gc
import math
import random
from tqdm import tqdm, trange
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model, Testing_ROUGE
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection


# Federated Summ with BERTSUM model

def Fed_Sum_BERTSUMEXT_without_Proto_Param(device,
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
    average_weight = np.array([float(i / training_dataset_size) for i in client_datasets_size_list])

    # Parameter Initialization
    λ = 0.5
    Q_bar = 0
    epsilon_bar = 0

    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]
    local_model_list = [0 for _ in range(num_clients_K)]

    clf_tuple = ()  # (idxs_users, clf_list)
    criterion = torch.nn.BCELoss(reduction='none')

    # 计算一共进行了多少个Batch的更新
    local_update_times_list = [int(math.ceil(size / param_dict['batch_size'])) for size in client_datasets_size_list]

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')

    # communication_round_I 是指总的通信轮次，论文里面就是最外层的T
    for iter_t in range(communication_round_I):
        # Client Selection 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=0,  # 掉队者在后面处理
            # style="FedAvg"
            style="FedProx"
        )

        client_average_one_sample_loss_in_epoch_list = [0 for _ in range(int(FL_fraction * num_clients_K))]

        if FL_drop_rate != 0:
            drop_count = math.ceil(FL_drop_rate * FL_fraction * num_clients_K)
            drop_idxs = random.sample(idxs_users, drop_count)
            connected_idxs = [id for id in idxs_users if id not in drop_idxs]


        # 下发模型
        for id in idxs_users:
            local_model_list[id] = copy.deepcopy(global_model)

        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        # Simulate Client Parallel
        for idxs_users_index, id in enumerate(idxs_users):
            epsilon_i_j_list, Q_i_j_list = [], []

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


            src_bak, labels_bak, segs_bak, clss_bak, mask_bak, mask_cls_bak = 0, 0, 0, 0, 0, 0
            # Local Training
            for epoch in range(algorithm_epoch_T):
                average_one_doc_loss_in_epoch = 0
                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                for batch_index, batch in enumerate(client_i_dataloader):
                    if batch_index == 0:
                        src_bak = batch['src'].to(device)[0]
                        labels_bak = batch['src_sent_labels'].to(device)[0]
                        segs_bak = batch['segs'].to(device)[0]
                        clss_bak = batch['clss'].to(device)[0]
                        mask_bak = batch['mask_src'].to(device)[0]
                        mask_cls_bak = batch['mask_cls'].to(device)[0]

                    src = batch['src'].to(device)
                    labels = batch['src_sent_labels'].to(device)
                    segs = batch['segs'].to(device)
                    clss = batch['clss'].to(device)
                    mask = batch['mask_src'].to(device)
                    mask_cls = batch['mask_cls'].to(device)
                    sent_label_flag = labels.gt(0.5)  # 尺寸【BatchSize, 批内最长句子数目】

                    # 分开每个sub_batch来计算，节约显存
                    sub_batch_size = 8
                    average_one_doc_loss_in_batch = 0
                    src_len = len(src)
                    sentence_loss_matrix = []

                    # 交叉熵训练
                    for i in range(0, src_len, sub_batch_size):
                        sbatch_size = src[i:i + sub_batch_size].shape[0]  # 获取当前批次的样本数量
                        # top_vec的尺寸：【sub_batch_size，最大输入字符数，768】
                        # sents_vec的尺寸：【sub_batch_size，批内最长句子数目，768】
                        top_vec, sents_vec = model.only_PLM_forward(src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    mask_cls[i:i + sbatch_size].reshape(sbatch_size,
                                                                                                        -1))

                        # pred_sent_scores的尺寸：【sub_batch_size，批内最长句子数目】
                        pred_sent_scores, tmp_mask = model.only_clf_forward(sents_vec, mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))

                        # print("sent_label_flag:")
                        # print(sent_label_flag)

                        # 注意，criterion函数并没有进行reduction操作，sub_batch_loss的尺寸：【sub_batch_size，sub_batch内最长句子数目】

                        sub_batch_loss = criterion(pred_sent_scores,
                                                   labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())

                        sentence_loss_matrix.append(sub_batch_loss)

                        # 为了避免爆显存，先回传loss
                        loss = torch.sum(sub_batch_loss) / src.shape[0]
                        # print("Loss: ", loss)
                        loss.backward()
                        torch.cuda.empty_cache()

                    # 计算整个batch的损失矩阵
                    sentence_loss_matrix = torch.concat(sentence_loss_matrix, 0)
                    L_data = float(torch.mean(torch.sum(sentence_loss_matrix,1), 0)) # 标准监督损失，先合成一个列，再做列平均得到标量。
                    # print("L_data:", L_data)

                    # 计算epsilon_{(i,j)}
                    epsilon_i_j = float(torch.mean(torch.sum(labels * sentence_loss_matrix, 1), 0))
                    # 计算Q_{(i,j)}
                    for column_index in range(labels.shape[1]):
                        # 先用sum函数把每个行的数据累和起来，再去判断到第几个列才能达到所需要的比例
                        if float( sum(sum(labels)[: column_index]) > (math.ceil(sum(sum(labels)))*λ) ):
                            Q_i_j = column_index
                            break

                    epsilon_i_j_list.append(epsilon_i_j / (algorithm_epoch_T * local_update_times_list[id]))
                    Q_i_j_list.append(Q_i_j / (algorithm_epoch_T * local_update_times_list[id]))

                    # 判断该batch是否为Prime样本
                    if (Q_i_j <= Q_bar) and (epsilon_i_j <= epsilon_bar):
                        # print("Normal batch")
                        # Normal样本，计算跳过该batch的概率
                        rho_i_j = (
                                ((iter_t + 1) / communication_round_I)
                                * ((epoch + 1) / algorithm_epoch_T)
                                * ((batch_index + 1) / local_update_times_list[id])
                        )
                        # 判断是否需要跳过该批样本
                        if rho_i_j >= random.uniform(0, 1):  # 要跳过
                            # loss += 0
                            model.zero_grad()
                        else:  # 不用跳过
                            # loss += L_data
                            average_one_doc_loss_in_batch += loss / src.shape[0]
                    else:
                        # prime样本，把标准监督损失通过梯度传去参数
                        # print("Prime batch")
                        # loss += L_data
                        average_one_doc_loss_in_batch += loss / src.shape[0]

                    average_one_doc_loss_in_epoch += average_one_doc_loss_in_batch / local_update_times_list[id]

                    # 梯度回传，释放显存
                    # print("The Batch loss is: ", loss)
                    # loss.backward()
                    # FedAvg算法一个batch就做一次更新
                    optimizer.step()
                    model.zero_grad()
                    torch.cuda.empty_cache()


                logger.info(f"### Communication Round: {iter_t + 1} / {communication_round_I}; "
                        f"Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_doc_loss_in_epoch}  ####")

            torch.cuda.empty_cache()
            # 计算样本跳过机制的数据
            for index, item in enumerate(epsilon_i_j_list):
                epsilon_bar += client_i_aggregation_weight * item
            for index, item in enumerate(Q_i_j_list):
                Q_bar += client_i_aggregation_weight * item

            # Upgrade the local model list
            local_model_list[id] = model.cpu()

            del model
            # del backup_model

        # Communicate

        # print("Q_bar:", Q_bar)
        # print("epsilon_bar:", epsilon_bar)
        logger.info(f"********** Communicate: {(iter_t + 1)} **********")

        # Global operation

        logger.info("********** Aggregation **********")

        theta_list = []
        clf_list = []
        for id in idxs_users:
            selected_model = local_model_list[id]
            theta_list.append(get_parameters(selected_model))
            clf_list.append(selected_model.ext_layer)

        theta_list = np.array(theta_list, dtype=object)
        # FedAvg新版论文的聚合权重是数据占比
        theta_avg = np.average(theta_list, axis=0, weights=[average_weight[j] for j in idxs_users]).tolist()
        # FedAvg旧版论文的聚合权重是平均
        # theta_avg = np.mean(theta_list, 0).tolist()
        set_parameters(global_model, theta_avg)

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
