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
from torchsummary import summary
import math
import torch.nn.functional as F


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


def compute_similarity(z1, z2):
    cos_sim = F.cosine_similarity(z1.view(-1, 768), z2.view(-1, 768))
    return cos_sim


def model_contrastive_loss(z_local, z_global, z_prev, tau=0.5):
    sim = compute_similarity(z_local, z_global)  # 计算当前本地模型与全局模型的相似度
    sim_prev = compute_similarity(z_local, z_prev)  # 计算当前本地模型与上一轮本地模型的相似度
    loss = -math.log(math.exp(sim / tau) / (math.exp(sim / tau) + math.exp(sim_prev / tau)))
    return loss


def get_final_feature(model, src, segs, clss, mask_src, mask_cls,device):
    model.to(device)
    _, _, rep = model(src, segs, clss, mask_src, mask_cls, output_represent=True)
    return rep


def MOON_BERTSUMEXT(device,
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

    # 这种方案有可能导致内存爆炸
    # Parameter Initialization
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]
    # 新的方案解决内存不足的问题
    backup_global_model = copy.deepcopy(global_model)

    # prev model
    # prev_params_dict = {}
    local_model_list = [1 for _ in range(num_clients_K)]
    prev_model_list = [1 for _ in range(num_clients_K)]

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

        # 抽到id的客户端才复制模型，以免爆内存
        for id in idxs_users:
            if local_model_list[id] == 1:
                local_model_list[id] = copy.deepcopy(backup_global_model)

        # # test model , to asure every time select the same client
        # idxs_users = [1,2]
        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        # Simulate Client Parallel
        for id in idxs_users:
            ## Local Initialization
            model = local_model_list[id]
            model.train()
            model.to(device)

            optimizer = BERTSUMEXT_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            if prev_model_list[id] == 1:
                prev_model_list[id] = copy.deepcopy(global_model)

            prev_model = copy.deepcopy(prev_model_list[id])

            ## Local Training
            for epoch in range(algorithm_epoch_T):
                epoch_total_loss = 0
                average_one_sample_loss_in_epoch = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # MOON算法中，一个batch就更新一次参数
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
                        sbatch_size = src[i:i + sub_batch_size].shape[0]  # 获取当前批次的样本数量
                        tmp_sent_scores, tmp_mask = model(src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                          mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))
                        # 注意，criterion函数并没有进行reduction操作，loss的尺寸：【sub_batch_size，sub_batch的大句子数目】
                        loss = criterion(tmp_sent_scores, labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())
                        # sub_batch_loss_list.append(loss)  #不要删，这行是表示先不回传小分批loss，最后再一整批loss回传，搭配下面

                        # print("=================loss before===============")
                        # print(loss) # Tensor(sub_batch_size,36)

                        # 为了避免爆显存，先回传loss
                        loss = torch.sum(loss) / src.shape[0]
                        # print("=================loss after===============")
                        # print(loss) # double值

                        # z_global,z_prev,z_cur指的是经过global，prev，cur三个模型的中间层输出（经过bert后的输出）
                        z_global = get_final_feature(global_model, src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                     segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                     clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                     mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                     mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1),device)

                        z_prev = get_final_feature(prev_model, src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                   segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                   clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                   mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                   mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1),device)

                        z_cur = get_final_feature(model, src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                  segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                  clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                  mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                  mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1),device)
                        loss_con = model_contrastive_loss(z_cur, z_global, z_prev, tau=0.5)
                        loss_con = 0
                        # loss要加上对比学习部分的loss_con
                        mu = 0.5
                        print("="*20)
                        print("loss:", loss)
                        print("loss_con",loss_con)
                        print("="*20)
                        loss = loss + mu * loss_con

                        average_one_sample_loss_in_batch += loss
                        loss.backward()
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

            ## Save current model and Upgrade the local model list
            prev_model_list[id] = copy.deepcopy(model)
            # prev_params_dict[id] = copy.deepcopy(model.state_dict())
            local_model_list[id] = model.cpu()
            del model

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
