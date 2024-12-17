import os
import json
import argparse
import pickle
from tool.logger import *
from tool.utils import check_and_make_the_path, str2bool
from experiment import Experiment
from moudle.experiment_setup import Experiment_Create_dataset


def Argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    # parser.add_argument("-algorithm", default='Perturbation', type=str)
    # parser.add_argument("-algorithm", default='FedProx', type=str)
    # parser.add_argument("-algorithm", default='FedRep', type=str)
    # parser.add_argument("-algorithm", default='FedAvg', type=str)
    parser.add_argument("-algorithm", default='FedSum', type=str)
    # parser.add_argument("-algorithm", default='FedDC', type=str)
    # parser.add_argument("-algorithm", default='Separate', type=str)
    # parser.add_argument("-algorithm", default='Centralized', type=str)
    # parser.add_argument("-algorithm", default='Ditto', type=str)
    # parser.add_argument("-algorithm", default='Scaffold', type=str)
    # parser.add_argument("-algorithm", default='FedNova', type=str)
    # parser.add_argument("-algorithm", default='FedSGD', type=str)

    parser.add_argument("-learning_rate", default=5e-3, type=float)  # 5e-5是follow邱锡鹏，实验发现[5e-2, 5e-3,…,5e-6]里面5e-3最好
    parser.add_argument("-optimize_method", default='sgd', type=str)
    parser.add_argument("-dataset", default='CNNDM', type=str)
    # parser.add_argument("-dataset", default='WikiHow', type=str)

    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-train_from", default='')
    # parser.add_argument("-tt", default="1", type=str, help="tt=1进行小批量运算，tt=0进行全量运算，详情在dataset.py")
    # parser.add_argument("-tt", default="0", type=str, help="tt=1进行小批量运算，tt=0进行全量运算，详情在dataset.py")
    parser.add_argument("-tt", default="few", type=str, help="tt=1进行小批量运算，tt=0进行全量运算，详情在dataset.py")

    parser.add_argument("-batch_size", default=32, type=int, help="batch size")  # 32 follow 邱锡鹏
    parser.add_argument("-cuda", default="1", type=str, help="cuda")

    args = parser.parse_args()
    param_dict = vars(args)
    param_dict["CUDA_VISIBLE_DEVICES"] = param_dict["cuda"]
    return param_dict


def main(dataset_name, algorithm, hypothesis, classifier_type, device, param_dict):
    ## Hyper-params
    # Common Hyper-params
    with open("./json/COMMON.json", "r") as f:
        temp_dict = json.load(f)
    param_dict.update(**temp_dict)
    # Dataset Hyper-params
    dataset_name_list = dataset_name.split(",")
    for dataset_name in dataset_name_list:
        dataset_name = dataset_name.strip()
        if os.path.exists(os.path.join("./json/dataset/", dataset_name + ".json")):
            with open(os.path.join("./json/dataset/", dataset_name + ".json"), "r") as f:
                temp_dict = json.load(f)
            param_dict.update(**temp_dict)
    # Algorithm Hyper-params
    if os.path.exists(os.path.join("./json/algorithm/", algorithm + ".json")):
        with open(os.path.join("./json/algorithm/", algorithm + ".json"), "r") as f:
            temp_dict = json.load(f)
        param_dict.update(**temp_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['CUDA_VISIBLE_DEVICES']

    import torch
    if "gpu" in device.lower():
        param_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment
    else:
        param_dict['device'] = "cpu"

    FL_drop_rate_list = [0]  # 设置掉线率
    epoch_T_communication_I_list = [(2, 5)]  # 本地走T个epoch后进行一次通信，共走T*I个epoch，每次聚合都做性能测试
    # split_strategy_list = ["Dirichlet01"]
    # split_strategy_list = ["Uniform"]

    # split_strategy_list = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Dirichlet8", "Uniform"]
    split_strategy_list = ["Uniform","Dirichlet01"]
    fraction_list = [0.2]
    # fraction_list = [0.1, 0.2, 0.3, 0.4]
    num_clients_K_list = [20]  # 设置客户端数目
    # num_clients_K_list = [20,30,40]  # 设置客户端数目
    # num_clients_K_list = [20,30,40,50]  # 设置客户端数目
    miu_list_4_FedProx = [1]
    param_dict['dataset_name'] = dataset_name
    param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis
    param_dict['classifier_type'] = classifier_type

    # Skipping the unnecessary loop
    # TODO 这里留给一些可以跳过部分循环的算法
    # if ("FedAvg".lower() in algorithm.lower()) or ("FedSGD".lower() in algorithm.lower()):
    # epoch_T_communication_I_list = [(2, 5), (4, 5), (6, 5)]  # 本地走T个epoch后进行一次通信，共走T*I个epoch，每次聚合都做性能测试
    if "FedProx".lower() in algorithm.lower():
        # miu_list_4_FedProx = [0.01, 0.1, 1]  # Dir05+20client下的实验效果0.01<0.1<1,所以选用1
        miu_list_4_FedProx = [1]
    if "Centralized".lower() in algorithm.lower():
        split_strategy_list = ["Uniform"]
        num_clients_K_list = [1]  # 设置客户端数目
        epoch_T_communication_I_list = [(2, 5), (4, 5), (6, 5)]  # 本地走T个epoch后进行一次通信，共走T*I个epoch，每次聚合都做性能测试

    # Test get label distribution
    epoch_T_communication_I_list = [(2,5)]

    # Serial number of experiment
    Experiment_NO = 1
    # total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list)
    total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list) * len(
        fraction_list) * len(num_clients_K_list) * len(miu_list_4_FedProx)

    # Create dataset
    logger.info("Creating dataset")
    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(param_dict)

    # 画图需要: 打印出数据集的中的标签分布
    # 只看数值
    # distribution = torch.sum(torch.stack([item['src_sent_labels'] for item in testing_dataset], 0), 0)
    # print(distribution)
    # 看比例
    distribution = torch.sum(torch.stack([item['src_sent_labels'] for item in testing_dataset], 0), 0)
    distribution_sum = sum(distribution)
    distribution_ratio = distribution / distribution_sum
    print(distribution_ratio)

    # Main Loop
    for split_strategy in split_strategy_list:
        for FL_drop_rate in FL_drop_rate_list:
            param_dict['FL_drop_rate'] = FL_drop_rate
            for algorithm_epoch_T, communication_round_I in epoch_T_communication_I_list:
                for fraction in fraction_list:
                    for num_clients_K in num_clients_K_list:
                        for miu in miu_list_4_FedProx:
                            param_dict['split_strategy'] = split_strategy
                            param_dict['num_clients_K'] = num_clients_K
                            param_dict['algorithm_epoch_T'] = algorithm_epoch_T
                            param_dict['communication_round_I'] = communication_round_I
                            param_dict['FL_fraction'] = fraction
                            param_dict['miu'] = miu

                            ################################################################################################
                            # Create the log
                            log_path = os.path.join("./log_path", param_dict['dataset_name'],
                                                    param_dict['split_strategy'],
                                                    param_dict['algorithm'],
                                                    param_dict['hypothesis'] + "(" + param_dict[
                                                        'classifier_type'] + "_Head)",
                                                    str(num_clients_K) + "Clients")
                            check_and_make_the_path(log_path)
                            log_path = os.path.join(log_path, str(Experiment_NO) + ".txt")
                            param_dict['log_path'] = log_path
                            file_handler = logging.FileHandler(log_path)
                            file_handler.setFormatter(formatter)
                            logger.addHandler(file_handler)
                            ################################################################################################
                            # Create the model path
                            model_path = os.path.join("./save_path", param_dict['dataset_name'],
                                                      param_dict['split_strategy'],
                                                      param_dict['algorithm'],
                                                      param_dict['hypothesis'] + "(" + param_dict[
                                                          'classifier_type'] + "_Head)",
                                                      str(num_clients_K) + "Clients")
                            check_and_make_the_path(model_path)
                            param_dict['model_path'] = model_path
                            for k in range(param_dict["num_clients_K"]):
                                _ = os.path.join(model_path, "client_" + str(k + 1))
                                check_and_make_the_path(_)
                            logger.info(f"FL_drop_rate_list = %s ;" % FL_drop_rate_list
                                        + "epoch_T_communication_I_list = %s ;\n" % epoch_T_communication_I_list
                                        + "split_strategy_list = %s ; " % split_strategy_list
                                        + "fraction_list = %s" % fraction_list)
                            logger.info(f"Experiment {Experiment_NO}/{total_Experiment_NO} setup finish")
                            param_dict['Experiment_NO'] = str(Experiment_NO)
                            ################################################################################################
                            # Parameter announcement
                            logger.info("Parameter announcement")
                            for para_key in list(param_dict.keys()):
                                if "_common" in para_key:
                                    continue
                                logger.info(f"****** {para_key} : {param_dict[para_key]} ******")
                            logger.info("-----------------------------------------------------------------------------")
                            ################################################################################################
                            # Experiment
                            # if Experiment_NO >= 9:
                            if True:
                                Experiment(param_dict, training_dataset, validation_dataset, testing_dataset)
                            Experiment_NO += 1
                            logger.removeHandler(file_handler)
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")


if __name__ == '__main__':
    param_dict = Argparse()
    main(dataset_name=param_dict['dataset'],
         algorithm=param_dict['algorithm'],
         hypothesis="BERTSUMEXT",
         # classifier_type="random_transformer",
         # classifier_type = "Transformer",
         classifier_type="linear",
         # classifier_type="baseline",
         device="gpu",
         param_dict=param_dict)
