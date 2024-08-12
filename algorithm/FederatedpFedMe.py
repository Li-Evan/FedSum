import torch
import numpy as np
import copy
import numpy as np
import copy
import math
import gc
from tool.logger import *
from tool.utils import get_parameters, get_tensor_parameters, set_parameters, save_model, Testing_ROUGE
from algorithm.Optimizers import BERTSUMEXT_Optimizer
from algorithm.client_selection import client_selection



def Fed_pFedMe_BERTSUMEXT(device,
                          global_model,
                          algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                          training_dataloaders,
                          training_dataset,
                          client_dataset_list,
                          param_dict,
                          testing_dataloader=None):

    if (param_dict["dataset_name"] == "Mixtape"):
        training_dataset_size = len(training_dataset)
    else:
        training_dataset_size = sum(len(i) for i in training_dataset)

    client_datasets_size_list = [len(item) for item in client_dataset_list]


    logger.info("Training process")

    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    criterion = torch.nn.BCELoss(reduction='none')

    for iter_t in range(communication_round_I):
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

        # 下发模型
        for id in idxs_users:
            local_model_list[id] = copy.deepcopy(global_model)

        for id in idxs_users:
            model = local_model_list[id]
            model.train()
            model.zero_grad()
            model.to(device)

            optimizer = BERTSUMEXT_Optimizer(algorithm="pFedMe",
                                             method="sgd",
                                             learning_rate=param_dict["plr"],
                                             max_grad_norm=0, lamda=param_dict["lamda"],
                                             mu=param_dict["mu"])
            optimizer.set_parameters(list(model.named_parameters()))

            client_i_dataloader = training_dataloaders[id]

            for epoch in range(algorithm_epoch_T):
                average_one_sample_loss_in_epoch = 0

                for batch_index, batch in enumerate(client_i_dataloader):
                    model.zero_grad()
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

                        # 为了避免爆显存，先回传loss
                        loss = torch.sum(loss) / src.shape[0]
                        average_one_sample_loss_in_batch += loss
                        loss.backward()
                        # torch.cuda.empty_cache()


                    average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                        client_datasets_size_list[id] / param_dict['batch_size'])
                    persionalized_model_bar=optimizer.step(get_tensor_parameters(model))

                    if (batch_index + 1) % param_dict["K"] == 0:
                        # update local weight after finding aproximate theta
                        for new_param, localweight in zip(persionalized_model_bar, model):
                            localweight.data = localweight.data - param_dict["lamda"] * param_dict["lr"] * (
                                    localweight.data - new_param.data)

                    # FedAvg算法一个batch就做一次更新
                    optimizer.step()
                    model.zero_grad()

                    del tmp_sent_scores, tmp_mask, src, labels, segs, clss, mask, mask_cls
                    gc.collect()

                logger.info(f"### Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss in Epoch: {average_one_sample_loss_in_epoch}  ####")

            # Upgrade the local model list
            local_model_list[id] = model.cpu()
            del model
            # torch.cuda.empty_cache()

        # Communicate
        logger.info(f"********** Communicate: {(iter_t + 1)} **********")

        # Global operation
        logger.info("********** Parameter aggregation **********")

        pre_param = np.array(get_parameters(global_model), dtype=object)

        new_global_params = (1 - param_dict["beta"]) * pre_param  #param_dict["beta"]

        # selected_training_dataset_size = 0
        # for id in idxs_users:
        #     selected_training_dataset_size += client_datasets_size_list[id]
        selected_training_dataset_size = sum(client_datasets_size_list[id] for id in idxs_users)

        # the weight of each selected client dependent on the numbers of dataset it has
        for id in idxs_users:
            selected_model = local_model_list[id]
            ratio = 1 / len(idxs_users)
            selected_param = get_parameters(selected_model)
            weighted_arr = np.array(selected_param, dtype=object) * ratio * param_dict["beta"]
            new_global_params += weighted_arr

        set_parameters(global_model, new_global_params)
        if (iter_t + 1) != param_dict['communication_round_I']:
            logger.info(f"Global model testing at Communication {(iter_t + 1)}")
            logger.info(f"########## Rouge_Testing Round: {iter_t + 1} / {communication_round_I}; ")
            Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model,
                          param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])

    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list
