import json
import csv
import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, random_split, Subset
from tool.utils import logger

np.random.seed(666)


def my_collate_fn(batch):
    src = torch.cat([b['src'].unsqueeze(0) for b in batch])
    tgt = torch.cat([b['tgt'].unsqueeze(0) for b in batch])
    src_sent_labels = torch.cat([b['src_sent_labels'].unsqueeze(0) for b in batch])
    segs = torch.cat([b['segs'].unsqueeze(0) for b in batch])
    clss = torch.cat([b['clss'].unsqueeze(0) for b in batch])
    mask_src = torch.cat([b['mask_src'].unsqueeze(0) for b in batch])
    mask_tgt = torch.cat([b['mask_tgt'].unsqueeze(0) for b in batch])
    mask_cls = torch.cat([b['mask_cls'].unsqueeze(0) for b in batch])
    src_txt = [b['src_txt'] for b in batch]
    tgt_txt = [b['tgt_txt'] for b in batch]
    tag = [b['tag'] for b in batch]
    return {
        "src": src,
        "tgt": tgt,
        "src_sent_labels": src_sent_labels,
        "segs": segs,
        "clss": clss,
        "mask_src": mask_src,
        "mask_tgt": mask_tgt,
        "mask_cls": mask_cls,
        "src_txt": src_txt,
        "tgt_txt": tgt_txt,
        'tag': tag
    }


def calculate_dataset_distribution(dataset, corpus_type, client_datasets=[]):
    dataset_ratios = {}
    dataset_sizes = {}  # A new dictionary to store the actual sizes

    if corpus_type == "test":
        dataset_ratios = {}
        dataset_sizes = len(dataset)
        for data_point in dataset:
            dataset_name = data_point["tag"]
            if dataset_name not in dataset_ratios:
                dataset_ratios[dataset_name] = 0
            dataset_ratios[dataset_name] += 1
        for dataset_name in dataset_ratios:
            dataset_ratios[dataset_name] /= dataset_sizes

        logger.info(f"Testing Dataset Ratios: {dataset_ratios}; Sizes: {dataset_sizes}")

    else:
        for i, client_dataset in enumerate(client_datasets):
            total_data_points = len(client_dataset)
            dataset_ratios[i] = {}
            dataset_sizes[i] = {}  # Initialize for this client

            for data_point_index in client_dataset.indices:
                dataset_name = dataset[data_point_index]["tag"]
                dataset_ratios[i][dataset_name] = dataset_ratios[i].get(dataset_name, 0) + 1
                dataset_sizes[i][dataset_name] = dataset_sizes[i].get(dataset_name, 0) + 1  # Counting the data points

            for dataset_name in dataset_ratios[i]:
                dataset_ratios[i][dataset_name] /= total_data_points  # Getting the ratio

        # Printing the dataset ratios and actual sizes
        if corpus_type == "train":
            logger.info("Test dataset Ratios: %s", dataset_ratios)
            logger.info("Test dataset Sizes: %s", dataset_sizes)

    # batch_idxs_example = [
    #     [1, 2, 3, 4, 5],  # data for client 0
    #     [6, 7, 8, 9, 10],  # data for client 1
    #     # ... (other clients)
    # ]


# 我要保存的batch_idxs数据太长了，不能保存在excel的不同sheet中。因此我想将其保存到csv文件中，请你据此修改我的代码重写save_to_csv(batch_idxs, num_clients)，load_from_csv(num_clients)
def save_to_csv(batch_idxs, num_clients):
    # Define the filename based on the number of clients
    file_name = f"./csv/batch_idxs_num_clients_{num_clients}.csv"

    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    # Saving data to a CSV file
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for client_data in batch_idxs:
            writer.writerow(client_data)

    print(f"Data successfully saved to {file_name}")


def save_to_excel(batch_idxs, num_clients):
    # batch_idxs_example = [
    #     [1, 2, 3, 4, 5],  # data for client 0
    #     [6, 7, 8, 9, 10],  # data for client 1
    #     # ... (other clients)
    # ]
    file_name = "./json/batch_idxs_and_num_clients."
    sheet_name = f"{num_clients}"  # 固定的工作表名称，如果需要，每次调用可以更改此处来添加新的工作表

    # Convert batch_idxs list of lists into a DataFrame
    df = pd.DataFrame(batch_idxs)

    # 如果文件不存在，创建一个新的Excel文件
    if not os.path.isfile(file_name):
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:  # 使用默认的写入模式，即'w'
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # 如果文件已经存在，则以追加模式打开文件并添加新工作表
        with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:  # 这里改变为追加模式
            # 为了避免重复的sheet名称，您可能需要动态地设置sheet名称
            # 这里为了简化示例，我假设每个新数据都需要一个新的sheet
            df.to_excel(writer, index=False, sheet_name=sheet_name)


def load_from_excel(num_clients):
    file_name = "./json/batch_idxs_and_num_clients.xlsx"
    sheet_name = f"{num_clients}"  # 使用num_clients来确定sheet名称

    # 检查文件是否存在
    if not os.path.isfile(file_name):
        print(f"文件{file_name}不存在。")
        return None  # 或者根据您的需要处理这个情况，比如引发异常

    try:
        # 从指定的工作表中读取数据
        with pd.ExcelFile(file_name) as xls:
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
            else:
                print(f"'{sheet_name}'工作表不存在于{file_name}中。")
                return None  # 或者根据您的需要处理这个情况，比如引发异常

        # 如果需要，可以在此进行其他数据处理，比如验证数据完整性等

        # 将DataFrame转换为列表的列表，方便后续处理
        batch_idxs = df.values.tolist()
        return batch_idxs

    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        # 根据您的错误处理策略，您可以选择引发异常或返回None
        return None


def load_from_csv(num_clients):
    # Define the filename based on the number of clients
    file_name = f"./csv/batch_idxs_num_clients_{num_clients}.csv"

    # Check if the file exists
    if not os.path.isfile(file_name):
        print(f"File {file_name} does not exist.")
        return None

    # Reading data back from the CSV file
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)

        # Reconstructing the data
        batch_idxs = []
        for row in reader:
            # Since CSV stores everything as strings, we need to convert data back to integers
            int_row = [int(item) for item in row]
            batch_idxs.append(int_row)

    return batch_idxs


def distribute_by_dataset(dataset, cumulative_sizes, dataset_tag_beta_map, num_clients):
    # A function to distribute indices by dataset type based on cumulative_sizes
    idx_ranges = [(0, cumulative_sizes[0])]
    for i in range(1, len(cumulative_sizes)):
        idx_ranges.append((cumulative_sizes[i - 1], cumulative_sizes[i]))

    distributed_idxs = {tag: list(range(start, end)) for (start, end), tag in
                        zip(idx_ranges, dataset_tag_beta_map.keys())}
    return distributed_idxs


def split_data_for_clients(dataset, num_clients, dataset_tag_beta_map, cumulative_sizes):
    distributed_idxs = distribute_by_dataset(dataset, cumulative_sizes, dataset_tag_beta_map, num_clients)

    # Initialize batch_idxs as a list of empty lists for each client
    batch_idxs = [[] for _ in range(num_clients)]

    for tag, indices in distributed_idxs.items():
        beta = dataset_tag_beta_map[tag]  # Get the beta for this dataset type

        min_size = 0
        while min_size < 1:  # Ensure at least one sample per client
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = proportions / proportions.sum()  # Normalize
            min_size = np.min(proportions * len(indices))

        # Split indices among clients based on calculated proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_idxs = np.split(indices, proportions)

        # Extend each client's list of indices with the new indices
        for client_idx_list, new_indices in zip(batch_idxs, split_idxs):
            client_idx_list.extend(new_indices.tolist())  # Convert numpy array to list before extending

    return batch_idxs


# def split_data_for_clients(dataset, num_clients, dataset_tag_beta_map, cumulative_sizes):
#     distributed_idxs = distribute_by_dataset(dataset, cumulative_sizes, dataset_tag_beta_map, num_clients)
#
#     # Initialize batch_idxs as a list of empty lists for each client
#
#     # clients_list = list(range(2,101))
#     clients_list = [8,16,32,64,128]
#     print(clients_list)
#     for i,item in enumerate(clients_list):
#         num_clients=clients_list[i]
#
#         batch_idxs = [[] for _ in range(num_clients)]
#
#         for tag, indices in distributed_idxs.items():
#             beta = dataset_tag_beta_map[tag]  # Get the beta for this dataset type
#
#             min_size = 0
#             while min_size < 10:  # Ensure at least one sample per client
#                 proportions = np.random.dirichlet(np.repeat(beta, num_clients))
#                 proportions = proportions / proportions.sum()  # Normalize
#                 #print("1")
#                 min_size = np.min(proportions * len(indices))
#
#             # Split indices among clients based on calculated proportions
#             proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
#             split_idxs = np.split(indices, proportions)
#
#             # Extend each client's list of indices with the new indices
#             for client_idx_list, new_indices in zip(batch_idxs, split_idxs):
#                 client_idx_list.extend(new_indices.tolist())  # Convert numpy array to list before extending
#         num_list=[len(clients_idex_list)for clients_idex_list in batch_idxs]
#         save_to_csv(batch_idxs,num_clients)
#         print(num_list)
#         # batch_idxs_dict = {num_clients: batch_idxs}
#         #
#         # save_to_json(batch_idxs_dict)
#     return batch_idxs

def get_FL_dataloader(param_dict, dataset, num_clients, split_strategy="Uniform",
                      do_train=True, batch_size=64,
                      do_shuffle=True, num_workers=0, corpus_type="test"
                      ):
    if "Dirichlet" in split_strategy:
        if param_dict["dataset"] == "Mixtape":
            # Map of dataset tags to their respective 'beta' values
            dataset_tag_beta_map = {
                "PubMed": 0.5,  # Example beta values, adjust as needed
                "GovernmentReport": 8,
                "WikiHow": 1,
                "CNNDM": 2,
                # Add other datasets with their betas here...
            }

            # shuffled_indices = np.random.permutation(len(dataset))
            # dataset = Subset(dataset, shuffled_indices)
            if param_dict["tt"] == "0":
                batch_idxs = load_from_csv(num_clients)
            else:
                batch_idxs = split_data_for_clients(dataset, num_clients, dataset_tag_beta_map,
                                                    dataset.cumulative_sizes)
        else:
            beta = 0.5
            if split_strategy == "Dirichlet01":
                beta = 0.1
            elif split_strategy == "Dirichlet05":
                beta = 0.5
            elif split_strategy == "Dirichlet1":
                beta = 1
            elif split_strategy == "Dirichlet8":
                beta = 8

            idxs = np.random.permutation(len(dataset))
            min_size = 0
            while min_size < 1:  # 每个客户至少拥有1个数据样本
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                proportions = proportions / proportions.sum()
                min_size = np.min(proportions * len(idxs))
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            batch_idxs = np.split(idxs, proportions)
        if do_train:
            client_datasets = [Subset(dataset, indices=batch_idxs[i]) for i in range(num_clients)]
            trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle,
                                       num_workers=num_workers, collate_fn=my_collate_fn) for ds in client_datasets]

            if param_dict["dataset"] == "Mixtape":
                calculate_dataset_distribution(dataset, corpus_type, client_datasets)

            return trainloaders, client_datasets

        else:
            calculate_dataset_distribution(dataset, corpus_type)

            testloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle,
                                    num_workers=num_workers, collate_fn=my_collate_fn)
            return testloader
    elif split_strategy == "Uniform":
        # Split training set into serval partitions to simulate the individual dataset
        partition_size = len(dataset) // num_clients
        # logger.info(partition_size)
        lengths = [partition_size] * num_clients

        remainder = len(dataset) - (partition_size * num_clients)
        lengths[-1] += remainder

        if do_train:
            client_datasets = random_split(dataset, lengths, torch.Generator().manual_seed(666))
            trainloaders = []
            for ds in client_datasets:
                # logger.info(len(ds))
                trainloaders.append(
                    DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers,
                               collate_fn=my_collate_fn))
            # logger.info("a"+str(len(trainloaders[0])))

            if param_dict["dataset"] == "Mixtape":
                calculate_dataset_distribution(dataset, corpus_type, client_datasets)
            return trainloaders, client_datasets


        else:
            calculate_dataset_distribution(dataset, corpus_type)
            testloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers,
                                    collate_fn=my_collate_fn)
            return testloader
    else:
        pass
