import json
import csv
import os

import torch
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


def calculate_dataset_distribution( dataset,  corpus_type,client_datasets=[]):

    dataset_ratios = {}
    dataset_sizes = {}  # A new dictionary to store the actual sizes


    if corpus_type=="test":
        dataset_ratios = {}
        dataset_sizes = len(dataset)
        for data_point in dataset:
            dataset_name = data_point["tag"]
            if dataset_name not in dataset_ratios:
                dataset_ratios[dataset_name] = 0
            dataset_ratios[dataset_name] += 1
        for dataset_name in dataset_ratios:
            dataset_ratios[dataset_name] /= dataset_sizes

        logger.info("test dataset Ratios: %s", dataset_ratios)
        logger.info("test dataset Sizes: %s", dataset_sizes)

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


# def load_from_json():
#     # Specify the file name
#     file_name = "batch_idxs_and_num_clients.json"
#
#     # Load the data from the JSON file
#     with open(file_name, "r") as file:
#         data = json.load(file)
#
#     return data["batch_idxs"], data["num_clients"]
def load_from_csv(num_clients):
    # Specify the file name
    file_name = "./json/batch_idxs_and_num_clients.csv"

    # Initialize an empty list to hold the batch indices
    batch_idxs = [[] for _ in range(num_clients)]

    # Load the data from the CSV file
    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)

        # Skip header
        next(reader, None)

        # Read data
        for row in reader:
            client_number = int(row[0])
            # Assuming indices are saved as a string of comma-separated values
            indices = list(map(int, row[1].strip('[]').split(',')))
            batch_idxs[client_number] = indices

    return batch_idxs

def save_to_csv(batch_idxs, num_clients):
    # Specify the file name
    file_name = "./json/batch_idxs_and_num_clients.csv"

    # Check if file exists to write header
    write_header = not os.path.exists(file_name)

    # Write the data to a CSV file
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if file didn't exist
        if write_header:
            writer.writerow(['Client Number', 'Batch Indices'])

        # Write data
        writer.writerow([num_clients, batch_idxs])




def load_from_csv(num_clients):
    # Specify the file name
    file_name = "./json/batch_idxs_and_num_clients.csv"

    # Initialize an empty list to hold the batch indices
    batch_idxs = [[] for _ in range(num_clients)]

    # Load the data from the CSV file
    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)

        # Skip header
        next(reader, None)

        # Read data
        for row in reader:
            client_number = int(row[0])
            # Assuming indices are saved as a string of comma-separated values
            indices = list(map(int, row[1].strip('[]').split(',')))
            if client_number==num_clients:
                batch_idxs = indices

    return batch_idxs
# def save_to_json(batch_idex_dict):
#     file_name=("./json/batch_idxs_and_num_clients.json")
#     with open(file_name,"a")as file:
#         json.dump(batch_idex_dict,file,indent=4)
def save_to_json(batch_idxs, num_clients):
    # Create a dictionary to hold the data
    data = {
        "batch_idxs": batch_idxs,
        "num_clients": num_clients
    }

    # Specify the file name
    file_name = "./json/batch_idxs_and_num_clients.json"

    # Write the data to a JSON file
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Data saved to {file_name}")

# def load_from_json(num_clients):
#     # Specify the file name
#     file_name = "./json/batch_idxs_and_num_clients.json"
#
#     # Load the data from the JSON file
#     with open(file_name, "r") as file:
#         batch_idxs_dict = json.load(file)
#
#     # Get the batch_idxs for the specified num_clients
#     batch_idxs = batch_idxs_dict.get(str(num_clients))  # Assuming num_clients is a string key
#
#     if batch_idxs is None:
#         raise ValueError(f"No batch_idxs found for num_clients: {num_clients}")
#
#     return batch_idxs


def distribute_by_dataset(dataset, cumulative_sizes, dataset_tag_beta_map, num_clients):
    # A function to distribute indices by dataset type based on cumulative_sizes
    idx_ranges = [(0, cumulative_sizes[0])]
    for i in range(1, len(cumulative_sizes)):
        idx_ranges.append((cumulative_sizes[i-1], cumulative_sizes[i]))

    distributed_idxs = {tag: list(range(start, end)) for (start, end), tag in zip(idx_ranges, dataset_tag_beta_map.keys())}
    return distributed_idxs

def split_data_for_clients(dataset, num_clients, dataset_tag_beta_map, cumulative_sizes):
    distributed_idxs = distribute_by_dataset(dataset, cumulative_sizes, dataset_tag_beta_map, num_clients)

    # Initialize batch_idxs as a list of empty lists for each client

    # clients_list = list(range(2,101))
    clients_list = [8,16,32,64,128]
    print(clients_list)
    for i,item in enumerate(clients_list):
        num_clients=clients_list[i]

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
        num_list=[len(clients_idex_list)for clients_idex_list in batch_idxs]
        save_to_csv(batch_idxs,num_clients)
        print(num_list)
        # batch_idxs_dict = {num_clients: batch_idxs}
        #
        # save_to_json(batch_idxs_dict)
    return batch_idxs

def  get_FL_dataloader(param_dict,dataset, num_clients, split_strategy="Uniform",
                      do_train=True, batch_size=64,
                      do_shuffle=True, num_workers=0,corpus_type="test"
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

            if param_dict["tt"]=="0":
                batch_idxs = load_from_csv(num_clients)
            else:
                batch_idxs = split_data_for_clients(dataset, num_clients, dataset_tag_beta_map, dataset.cumulative_sizes)

        else:
            beta = 0.5
            if split_strategy=="Dirichlet05":
                beta = 0.5
            elif split_strategy=="Dirichlet8":
                beta = 8
            elif split_strategy=="Dirichlet128":
                beta = 128

            idxs = np.random.permutation(len(dataset))
            min_size = 0
            while min_size < 1:
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
                calculate_dataset_distribution(dataset,  corpus_type, client_datasets)

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

            if param_dict["dataset"]=="Mixtape":

                calculate_dataset_distribution(dataset,  corpus_type,client_datasets)
            return trainloaders, client_datasets


        else:
            calculate_dataset_distribution( dataset,  corpus_type)
            testloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers, collate_fn=my_collate_fn)
            return testloader
    else:
        pass

