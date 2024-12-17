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

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor


def model_perturb(operated_model, mask_rate):
    ori_param_dict = {param_name: param_value for param_name, param_value in operated_model.named_parameters()}

    with torch.no_grad():
        new_params = {}
        for param_name in ori_param_dict:
            new_params[param_name] = mask_input_with_mask_rate(ori_param_dict[param_name], mask_rate, use_rescale=False,
                                                               mask_strategy="random")
            ori_param_dict[param_name].data.copy_(new_params[param_name])

    for param in operated_model.parameters():
        param.requires_grad = False

    return operated_model

def get_PDT(num_clients_K, semantic_portrait_list):
    portrait_distance_table = [[0 for _ in range(num_clients_K)] for _ in range(num_clients_K)]

    for i in range(num_clients_K):
        portrait_i = semantic_portrait_list[i]
        for j in range(i, num_clients_K):
            portrait_j = semantic_portrait_list[j]
            dist = torch.dist(portrait_i, portrait_j)  # 两个语义画像之间的L2距离（也就是欧氏距离）
            portrait_distance_table[i][j] = math.tanh(dist)
            portrait_distance_table[j][i] = math.tanh(dist)

    return portrait_distance_table

def Fed_Sum_BERTSUMEXT(device,
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

    # 获取各客户的初始语义表示
    global_model.to(device)
    semantic_portrait_list = []  # 每一个语义画像 =【标签0原型，标签1原型】
    for id in range(num_clients_K):
        label_0_feature_list = []
        label_1_feature_list = []
        # 每个client随机取一个batch，0原型、1原型的堆叠矩阵作为语义画像
        with torch.no_grad():
            client_i_dataloader = training_dataloaders[id]
            for batch_index, batch in enumerate(client_i_dataloader):
                src = batch['src'].to(device)
                labels = batch['src_sent_labels'].to(device)
                segs = batch['segs'].to(device)
                clss = batch['clss'].to(device)
                mask = batch['mask_src'].to(device)
                mask_cls = batch['mask_cls'].to(device)
                sub_batch_size = 32
                src_len = len(src)
                for i in range(0, src_len, sub_batch_size):
                    sbatch_size = src[i:i + sub_batch_size].shape[0]  # 获取当前批次的样本数量
                    # top_vec的尺寸：【src_len，最大输入字符数，768】
                    # sents_vec的尺寸：【src_len，批内最长句子数目，768】
                    top_vec, sents_vec = global_model.only_PLM_forward(
                        src[i:i + sbatch_size].reshape(sbatch_size, -1),
                        segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                        clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                        mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                        mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))

                    # 添加类原型素材
                    sent_label_flag = labels.gt(0.5)
                    # torch.Size([sub_batch_size, 批内最长词数目, 768])
                    # print(sents_vec.shape)
                    for doc_index in range(src_len):
                        for sent_index, sent_feature in enumerate(sents_vec[doc_index]):
                            # sent_feature尺寸：【768，】
                            if torch.all(sent_feature == 0):  # 排除用于对齐输出长度的tensor
                                continue

                            if sent_label_flag[i:i + sbatch_size][doc_index][sent_index]:
                                # logger.info("Append label 1 feature! ")
                                label_1_feature_list.append(sent_feature)
                            else:
                                # logger.info("Append label 0 feature! ")
                                label_0_feature_list.append(sent_feature)
                if (len(label_1_feature_list) != 0) and (len(label_0_feature_list) != 0):
                    break

            del top_vec, sents_vec, sent_label_flag, src, labels, segs, clss, mask, mask_cls
            gc.collect()

            # 三种prototype尺寸：【768，】
            label_0_prototype = torch.stack(label_0_feature_list, dim=0).mean(dim=0)
            try:
                label_1_prototype = torch.stack(label_1_feature_list, dim=0).mean(dim=0)
            except Exception:
                label_1_prototype = label_0_prototype * 0
            # client_semantic_matrix尺寸：【3,768】
            semantic_portrait = torch.stack([label_0_prototype, label_1_prototype], dim=0)

        semantic_portrait_list.append(semantic_portrait)
    global_model.to("cpu")

    # 获取各客户的初始语义L2距离（也就是欧氏距离）
    portrait_distance_table = get_PDT(num_clients_K, semantic_portrait_list)

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')

    global_label_0_prototype_list = []  # 存放每个客户的 0标签prototype，这个里面的内容是乘了数据量多项式分布权重的
    raw_global_label_0_prototype_list = []  # 存放每个客户的  0标签prototype，这里没有乘任何内容
    global_label_1_prototype_list = []  # 存放每个客户的 1标签prototype，这个里面的内容是乘了数据量多项式分布权重的
    raw_global_label_1_prototype_list = []  # 存放每个客户的  1标签prototype，这里没有乘任何内容
    global_predict_0_prototype_list = []
    global_predict_1_prototype_list = []

    global_predict_0_feature_list = []
    global_predict_1_feature_list = []
    global_label_0_feature_list = []
    global_label_1_feature_list = []

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

                # Semantic-guided Inter-client knowledge transfer
                # 学习其他CLF中的知识，提高表征能力
                if len(clf_tuple) != 0:
                    L_pm = 0
                    for batch_index, batch in enumerate(client_i_dataloader):
                        src = batch['src'].to(device)
                        labels = batch['src_sent_labels'].to(device)
                        segs = batch['segs'].to(device)
                        clss = batch['clss'].to(device)
                        mask = batch['mask_src'].to(device)
                        mask_cls = batch['mask_cls'].to(device)
                        sub_batch_size = 8
                        src_len = len(src)
                        last_idxs_users, last_clf_list = clf_tuple
                        len_last_clf_list = len(last_clf_list)
                        for i in range(0, src_len, sub_batch_size):
                            sbatch_size = src[i:i + sub_batch_size].shape[0]  # 获取当前批次的样本数量
                            for num, clf in enumerate(last_clf_list):
                                # 获取当前客户与其他CLF来源客户的语义画像距离
                                other_id = last_idxs_users[num]
                                distance = portrait_distance_table[id][other_id]
                                # 以距离加权损失，多补充缺少的语义信息，少补充已有的语义信息。
                                gradient_weighted = float(distance)
                                clf.to(device)
                                # 显存不够，只能重新产生一次计算图
                                _, sents_vec = model.only_PLM_forward(
                                    src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                    segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                    clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                    mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                    mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))
                                # tmp_sent_scores的尺寸：【sub_batch_size，批内最长句子数目】
                                tmp_sent_scores = clf(sents_vec, mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))
                                # 注意，criterion函数并没有进行reduction操作，sub_batch_loss的尺寸：【sub_batch_size，sub_batch内最长句子数目】
                                sub_batch_loss = criterion(tmp_sent_scores,
                                                           labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())
                                loss = gradient_weighted * torch.sum(sub_batch_loss) / (len_last_clf_list * src.shape[0])
                                L_pm += loss
                                loss.backward()
                                clf.cpu()
                            break

                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()
                        model.zero_grad()
                        # break
                    # print("L_pm: ", L_pm)


                logger.info(f"### Communication Round: {iter_t + 1} / {communication_round_I}; "
                        f"Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_doc_loss_in_epoch}  ####")

            torch.cuda.empty_cache()
            client_i_feature_list = []
            client_i_predict_0_feature_list = []
            client_i_predict_1_feature_list = []
            client_i_label_0_feature_list = []
            client_i_label_1_feature_list = []
            # 添加原型素材
            with torch.no_grad():
                for batch_index, batch in enumerate(client_i_dataloader):
                    src = batch['src'].to(device)
                    labels = batch['src_sent_labels'].to(device)
                    segs = batch['segs'].to(device)
                    clss = batch['clss'].to(device)
                    mask = batch['mask_src'].to(device)
                    mask_cls = batch['mask_cls'].to(device)
                    sent_label_flag = labels.gt(0.5)  # 尺寸【BatchSize, 批内最长句子数目】

                    # 分开每个sub_batch来计算，节约显存
                    sub_batch_size = 8
                    src_len = len(src)
                    for i in range(0, src_len, sub_batch_size):

                        sbatch_size = src[i:i + sub_batch_size].shape[0]  # 获取当前批次的样本数量
                        # top_vec的尺寸：【sub_batch_size，最大输入字符数，768】
                        # sents_vec的尺寸：【sub_batch_size，批内最长句子数目，768】
                        top_vec, sents_vec = model.only_PLM_forward(src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                                    mask_cls[i:i + sbatch_size].reshape(sbatch_size,-1))

                        # pred_sent_scores的尺寸：【sub_batch_size，批内最长句子数目】
                        pred_sent_scores, tmp_mask = model.only_clf_forward(sents_vec, mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))


                        for pred_doc_index, pred_doc in enumerate(pred_sent_scores):
                            for sent_index, sent_flag in enumerate(pred_doc.gt(0.5)):
                                sent_feature = sents_vec[pred_doc_index][sent_index]
                                # 添加超原型素材
                                if torch.all(sent_feature == 0):  # 排除用于对齐输出长度的tensor
                                    continue
                                client_i_feature_list.append(sent_feature)  # 局部超原型素材

                                # 根据预测结果添加预测原型素材
                                if sent_flag:
                                    # logger.info("Append predict 1 feature! ")
                                    client_i_predict_1_feature_list.append(sent_feature)
                                else:
                                    # logger.info("Append predict 0 feature! ")
                                    client_i_predict_0_feature_list.append(sent_feature)

                                # 添加类原型素材
                                if sent_label_flag[i:i + sbatch_size][pred_doc_index][sent_index]:
                                    # logger.info("Append label 1 feature! ")
                                    client_i_label_1_feature_list.append(sent_feature)
                                else:
                                    # logger.info("Append label 0 feature! ")
                                    client_i_label_0_feature_list.append(sent_feature)



                    model.zero_grad()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
            # print("len(client_i_predict_1_feature_list) ：", len(client_i_predict_1_feature_list))
            # print("len(client_i_predict_0_feature_list) ：", len(client_i_predict_0_feature_list))
            # print("len(client_i_label_1_feature_list) ：", len(client_i_label_1_feature_list))
            # print("len(client_i_label_0_feature_list) ：", len(client_i_label_0_feature_list))

            # 衡量类原型
            if len(client_i_label_0_feature_list) != 0:
                client_i_label_0_prototype = torch.stack(client_i_label_0_feature_list, dim=0).mean(dim=0)
                semantic_portrait_list[id][0] = client_i_label_0_prototype  # 更新本轮通信过的client的语义画像
            if len(client_i_label_1_feature_list) != 0:
                client_i_label_1_prototype = torch.stack(client_i_label_1_feature_list, dim=0).mean(dim=0)
                semantic_portrait_list[id][1] = client_i_label_1_prototype  # 更新本轮通信过的client的语义画像
            # 衡量预测原型
            if len(client_i_predict_0_feature_list) != 0:
                client_i_predict_0_prototype = torch.stack(client_i_predict_0_feature_list, dim=0).mean(dim=0)
            if len(client_i_predict_1_feature_list) != 0:
                client_i_predict_1_prototype = torch.stack(client_i_predict_1_feature_list, dim=0).mean(dim=0)


            # 计算客户的 预测原型
            if len(client_i_predict_0_feature_list) != 0:
                global_predict_0_feature_list.append(client_i_aggregation_weight * client_i_predict_0_prototype)
            if len(client_i_predict_1_feature_list) != 0:
                global_predict_1_feature_list.append(client_i_aggregation_weight * client_i_predict_1_prototype)
            # 计算客户的 类原型
            if len(client_i_label_0_feature_list) != 0:
                raw_global_label_0_prototype_list.append(client_i_label_0_prototype)
                global_label_0_feature_list.append(client_i_aggregation_weight * client_i_label_0_prototype)
            if len(client_i_label_1_feature_list) != 0:
                raw_global_label_1_prototype_list.append(client_i_label_1_prototype)
                global_label_1_feature_list.append(client_i_aggregation_weight * client_i_label_1_prototype)

            # 计算原型损失
            (label_0_feature_gap, label_1_feature_gap,
             local_predict_01_feature_gap) = 0, 0, 0
            # 计算原型Gap
            with torch.no_grad():
                # 全局-本地类原型的差异
                if (len(global_label_0_prototype_list) != 0) and (len(client_i_label_0_feature_list) != 0):
                    label_0_feature_gap = torch.norm((global_label_0_prototype_list[-1] - client_i_label_0_prototype), p=2)
                if (len(global_label_1_prototype_list) != 0) and (len(client_i_label_1_feature_list) != 0):
                    label_1_feature_gap = torch.norm((global_label_1_prototype_list[-1] - client_i_label_1_prototype), p=2)

                # 希望提高预测的差异常度，拉大预测原型之间差距
                if (len(client_i_predict_0_feature_list) != 0) and (len(client_i_predict_1_feature_list) != 0):
                    try:
                        local_predict_01_feature_gap = torch.norm((client_i_predict_1_prototype - client_i_predict_0_prototype), p=2)
                    except Exception:
                        logger.info("local_predict_01_feature_gap error, set it as 0")
                        local_predict_01_feature_gap = 0


            weighted_list = [1, 1, -1]
            gap_list = [label_0_feature_gap, label_1_feature_gap,
                        local_predict_01_feature_gap]
            # print("gap_list:", gap_list)
            # 重新产生一次计算图
            top_vec, sents_vec = model.only_PLM_forward(src_bak.reshape(1, -1),
                                                        segs_bak.reshape(1, -1),
                                                        clss_bak.reshape(1, -1),
                                                        mask_bak.reshape(1, -1),
                                                        mask_cls_bak.reshape(1, -1))
            for param in model.ext_layer.parameters():
                param.requires_grad = False
            tmp_sent_scores, _ = model.only_clf_forward(sents_vec, mask_cls_bak.reshape(1, -1))
            loss = torch.sum(criterion(tmp_sent_scores, labels_bak.reshape(1, -1).float())) * 0
            L_po = 0
            for index, weight in enumerate(weighted_list):
                # 原型Gap加权得到原型损失，加入到loss中
                L_po += weight * gap_list[index]
            # print("L_po:", L_po)
            loss += L_po
            loss.backward()
            optimizer.step()
            model.zero_grad()
            for param in model. parameters():
                param.requires_grad = True

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

        if FL_drop_rate != 0:
            logger.info("********** Straggler approximation **********")
            # 通过目前连接的客户参数，根据语义相似程度，模拟掉队者的参数
            for id in drop_idxs:
                drop_model = copy.deepcopy(global_model)
                approximation_theta_list = []
                similarity_summ = 0
                for conn_id in connected_idxs:
                    conn_theta = get_parameters(local_model_list[id])
                    distance = portrait_distance_table[id][conn_id]
                    similarity = math.exp(-distance)
                    similarity_summ += similarity
                    approximation_theta = [weight * similarity for weight in conn_theta]
                    approximation_theta_list.append(approximation_theta)
                for theta_list_index in range(len(approximation_theta_list)):
                    tmp_theta = approximation_theta_list[theta_list_index]
                    new_approximation_theta = [weight / similarity_summ for weight in tmp_theta]
                    approximation_theta_list[theta_list_index] = new_approximation_theta

                approximation_theta_list = np.array(approximation_theta_list, dtype=object)
                # approximation_theta_sum = np.average(approximation_theta_list, axis=0, weights=[1 for _ in connected_idxs]).tolist()
                approximation_theta_sum = np.sum(approximation_theta_list, axis=0).tolist()
                # 更新掉队者参数
                set_parameters(drop_model, approximation_theta_sum)
                local_model_list[id] = drop_model

        logger.info("********** Aggregation **********")
        (global_predict_0_prototype, global_predict_1_prototype,
         global_label_0_prototype, global_label_1_prototype) = 0, 0, 0, 0
        # 全局原型聚合
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

        # 对收集的CLF模块加噪防止过拟合
        logger.info("********** Perturb CLF moudle **********")
        mask_rate_γ = 0.3
        for index, clf in enumerate(clf_list):
            new_clf = model_perturb(clf, mask_rate_γ)
            clf_list[index] = new_clf
        # 更新CLF列表
        clf_tuple = (idxs_users, clf_list)

        # 更新PDT
        portrait_distance_table = get_PDT(num_clients_K, semantic_portrait_list)

        # 通信后检测性能
        if (iter_t + 1) != param_dict['communication_round_I']:
            logger.info(f"Global model testing at Communication {(iter_t + 1)}")
            logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; ")
            Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model,
                          param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


        # Save model for test label distribution
        # import os
        # set_parameters(global_model, theta_avg)
        # save_dir = f'save_path'
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"global_fedsum_{iter_t}.pth")
        # torch.save(global_model.state_dict(), save_path)
        # print("Save FedSum global model in iter ", iter_t + 1)

        # Save model
        # TODO:现在是若干个通信轮次之后统一保存一次global和一次client，有必要的话可以改成在客户端的迭代里面保存，但感觉这个问题不大
        # if (iter_t) % param_dict["save_checkpoint_rounds"] == 0 and iter_t != 0:
        #     save_model(param_dict, global_model, local_model_list, iter_t)

    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list
