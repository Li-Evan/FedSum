# This code is for the future work
import torch
import copy
import math
import numpy as np
import gc
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model,save_model_sepa, Testing_ROUGE
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection

# Federated Average with BERTSUM
def ST_Bertsum(device,
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

    # Parameter Initialization
    local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)]
    criterion = torch.nn.BCELoss(reduction='none')

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')

    # Simulate Client Parallel
    for id in range(num_clients_K):
        model = local_model_list[id]
        model.train()
        model.to(device)

        optimizer = BERTSUMEXT_Optimizer(
            method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        optimizer.set_parameters(list(model.named_parameters()))

        client_i_dataloader = training_dataloaders[id]

        # Local Training
        # for epoch in tqdm(range(algorithm_epoch_T), desc="Epoch"):
        for epoch in range(algorithm_epoch_T*communication_round_I):
            epoch_total_loss = 0
            average_one_sample_loss_in_epoch = 0

            # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
            # FedAvg算法中，一个batch就更新一次参数
            for batch_index, batch in enumerate(client_i_dataloader):  # 只需要直接执行完一个batch后break就行
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
                average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(client_datasets_size_list[id] / param_dict['batch_size'])


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

            # if epoch + 1 % (algorithm_epoch_T / 2) == 0:
            #     Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, model,
            #                   algorithm_epoch_T, communication_round_I)

            logger.info(f"##########SeparateTraining"
                        f"Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_sample_loss_in_epoch}  ####")

        # Upgrade the local model list
        local_model_list[id] = model.cpu()
        del model
        # torch.cuda.empty_cache()

    logger.info("Training finish, return global model and local model list")
    return local_model_list
