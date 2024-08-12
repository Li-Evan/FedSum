import os
import torch
from tool.logger import *
from tool.utils import check_and_make_the_path, get_parameters, test_bertscore
from moudle.experiment_setup import Experiment_Create_dataloader, Experiment_Create_model, Experiment_Reload_model

from algorithm.FederatedProximal import Fed_Prox_BERTSUMEXT
from algorithm.FederatedpFedMe import Fed_pFedMe_BERTSUMEXT
from algorithm.FederatedSuPerFed import Fed_SuPerFed_BERTSUMEXT
from algorithm.Scaffold import Scaffold_BERTSUMEXT
from algorithm.SeparateTraining import ST_Bertsum
from algorithm.FederatedRep import Fed_Rep_BERTSUMEXT
from algorithm.FederatedAverage import Fed_AVG_BERTSUMEXT
from algorithm.FederatedSGD import Fed_SGD_BERTSUMEXT
# from algorithm.FederatedPerturb import Fed_AVG_PERTURB_BERTSUMEXT
from algorithm.FederatedSum import Fed_Sum_BERTSUMEXT
from algorithm.FederatedSum_without_Proto_Param import Fed_Sum_BERTSUMEXT_without_Proto_Param
from algorithm.FederatedProto import Fed_PROTO_BERTSUMEXT

from algorithm.FederatedNova import Fed_Nova_BERTSUMEXT
from algorithm.Ditto import Ditto_BERTSUMEXT
from algorithm.FederatedDaisyChaining import Fed_DC_BERTSUMEXT
from tool.utils import Testing_ROUGE, Testing_ROUGE_separate



def Experiment_SeparateTraining(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 训练
    client_model_list = ST_Bertsum(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 测试
    logger.info("Client models testing")

    Testing_ROUGE_separate(param_dict, param_dict['device'], testing_dataloader, client_model_list,
                           param_dict['algorithm_epoch_T'])



def Experiment_Giant(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                     testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 训练
    updated_global_model, client_model_list = Fed_Giant(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    # _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    # check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    # torch.save(client_model, _)

    # 测试
    logger.info("Global model testing")
    # updated_global_model = torch.load(param_dict['model_path'] + "/global_model.pkl")
    # testing_model = torch.load(param_dict['model_path'] + "/global_model.pkl")
    # for testing_model in client_model_list.append(updated_global_model):
    for testing_model in [updated_global_model]:
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, testing_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_Average(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    if torch.cuda.is_available():
        logger.info("Initial Global model testing")
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
        torch.cuda.empty_cache()



    # 训练
    updated_global_model, client_model_list = Fed_AVG_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_Average_test_distribution(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    if torch.cuda.is_available():
        logger.info("Initial Global model testing")
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
        torch.cuda.empty_cache()

    # 训练
    model_path = os.path.join("save_path","global_fedavg_0.pth")
    updated_global_model = torch.load(model_path)


    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_SGD(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_SGD_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_Nova(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_Nova_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_Average_Perturbation(param_dict, global_model, training_dataloaders, training_dataset,
                                              client_dataset_list,
                                              testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_AVG_PERTURB_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])

def Experiment_Federated_Prox(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                              testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_Prox_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # TODO

    # 测试
    logger.info("Global model testing")
    # testing_model = torch.load(param_dict['model_path'] + "/global_model.pkl")
    # for testing_model in client_model_list.append(updated_global_model):
    for testing_model in [updated_global_model]:
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, testing_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_pFedMe(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_pFedMe_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # TODO

    # 测试
    logger.info("Global model testing")
    # testing_model = torch.load(param_dict['model_path'] + "/global_model.pkl")
    # for testing_model in client_model_list.append(updated_global_model):
    for testing_model in [updated_global_model]:
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, testing_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_SuPerFed(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                  testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_SuPerFed_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # TODO

    # 测试
    # TODO


def Experiment_Scaffold(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                        testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Scaffold_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储

    # 测试
    logger.info("Global model testing")
    # testing_model = torch.load(param_dict['model_path'] + "/global_model.pkl")
    # for testing_model in client_model_list.append(updated_global_model):
    for testing_model in [updated_global_model]:
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, testing_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_Rep(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                             testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_Rep_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #     _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #     check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #     torch.save(client_model, _)

    # 测试
    logger.info("Global model testing")
    # testing_model = torch.load(param_dict['model_path'] + "/global_model.pkl")
    # for testing_model in client_model_list.append(updated_global_model):
    for testing_model in [updated_global_model]:
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, testing_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_Daisy_Chaining(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    # if int(param_dict['Experiment_NO']) == 1:
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_DC_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])

def Experiment_Federated_Sum(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_Sum_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment_Federated_Sum_No_Proto_No_Param(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_Sum_BERTSUMEXT_without_Proto_Param(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])



def Experiment_Federated_Proto(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    if torch.cuda.is_available():
        logger.info("Initial Global model testing")
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
        torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Fed_PROTO_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 存储
    # logger.info("Global model Saving")
    # check_and_make_the_path(param_dict['model_path'])
    # torch.save(updated_global_model, param_dict['model_path'] + "/global_model.pkl")
    # logger.info("Client Models Saving")
    # for client_id, client_model in enumerate(client_model_list):
    #      _ = os.path.join(param_dict['model_path'], "client_" + str(client_id + 1) + "/model.pkl")
    #      check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
    #      torch.save(client_model, _)

    # 测试
    logger.info("Updated Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, updated_global_model,
                  param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])




def Experiment_Ditto(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                     testing_dataloader, validation_dataloaders=None):
    device = param_dict['device']

    # 验证
    logger.info("Initial Global model testing")
    Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, global_model, 0, 0)
    torch.cuda.empty_cache()

    # 训练
    updated_global_model, client_model_list = Ditto_BERTSUMEXT(
        device,
        global_model,
        param_dict['algorithm_epoch_T'],
        param_dict['num_clients_K'],
        param_dict['communication_round_I'],
        param_dict['FL_fraction'],
        param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader
    )
    logger.info("-----------------------------------------------------------------------------")

    # 测试
    logger.info("Global model testing")
    # testing_model = torch.load(param_dict['model_path'] + "/global_model.pkl")
    # for testing_model in client_model_list.append(updated_global_model):
    for testing_model in [updated_global_model]:
        Testing_ROUGE(param_dict, param_dict['device'], testing_dataloader, testing_model,
                      param_dict['algorithm_epoch_T'], param_dict['communication_round_I'])


def Experiment(param_dict, training_dataset, validation_dataset, testing_dataset):
    # # Create dataset
    # logger.info("Creating dataset")
    # training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(param_dict)

    # Create dataloader
    logger.info("Creating dataloader")
    # training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(
    #     param_dict, training_dataset, validation_dataset, testing_dataset, param_dict['split_strategy'])
    training_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(
        param_dict, training_dataset, validation_dataset, testing_dataset, param_dict['split_strategy'])

    # Model Construction
    # 为了避免过多的随机性影响，尽量保证在同一个初始的模型开始训练
    if "linear" in param_dict['classifier_type']:
        global_init_model_path = r"./save_path/global_model_init.pt"
        global_model = Experiment_Create_model(param_dict)
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            global_model.load_state_dict(torch.load(global_init_model_path).state_dict())
        global_model.bert.finetune=True
    elif "baseline" in param_dict['classifier_type']:
        global_init_model_path = r"./save_path/toy_global_model_init.pt"
        global_model = Experiment_Create_model(param_dict)
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            global_model.load_state_dict(torch.load(global_init_model_path).state_dict())
        global_model.bert.finetune = True
    else:
        raise AssertionError

    # if os.path.exists(param_dict["current_model"]):
    #     global_model.load_state_dict(torch.load(param_dict["current_model"]).state_dict())
    # global_model.to(param_dict['device'])
    logger.info("-----------------------------------------------------------------------------")
    print(f'Algorithm Name: {param_dict["algorithm"]}')

    # Perturbation
    if ("Perturbation" in param_dict['algorithm']):
        logger.info("~~~~~~ Algorithm: Perturbation ~~~~~~")
        Experiment_Federated_Average_Perturbation(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader
        )
    # SeparateTraining
    elif ("Separate" in param_dict["algorithm"]) or ("separate" in param_dict["algorithm"]) or (
            "sepa" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: SeparateTraining ~~~~~~")
        Experiment_SeparateTraining(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader

        )
    # CentralizedTraining
    elif ("Centralized" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: CentralizedTraining ~~~~~~")
        Experiment_SeparateTraining(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader

    )
    # Federated SGD
    elif ("FederatedSGD" in param_dict["algorithm"]) or ("FedSGD" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated SGD ~~~~~~")
        Experiment_Federated_SGD(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader
        )

    # Federated Average
    elif ("FederatedAverage" in param_dict["algorithm"]) or ("FedAvg" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Average ~~~~~~")
        Experiment_Federated_Average(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader
        )
    elif ("FedAvgGetLabel" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Average Get Label~~~~~~")
        Experiment_Federated_Average_test_distribution(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader
        )

    # Federated Nova
    elif ("FederatedNova" in param_dict["algorithm"]) or ("FedNova" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Nova ~~~~~~")
        Experiment_Federated_Nova(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader
        )

    # Federated Prox
    elif ("FederatedProximal" in param_dict["algorithm"]) or ("FedProx" in param_dict["algorithm"]) or (
            "fedprox" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Proximal ~~~~~~")
        Experiment_Federated_Prox(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                  testing_dataloader)
    # pFedMe
    elif ("pFedMe" in param_dict["algorithm"]) or ("pfedme" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: pFedMe ~~~~~~")
        Experiment_Federated_pFedMe(param_dict, global_model, training_dataloaders, training_dataset,
                                    client_dataset_list,
                                    testing_dataloader)
    # SuPerFed
    elif ("SuPerFed" in param_dict["algorithm"]) or ("superfed" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: SuPerFed ~~~~~~")
        Experiment_Federated_SuPerFed(param_dict, global_model, training_dataloaders, training_dataset,
                                      client_dataset_list, testing_dataloader)
    # SCAFFOLD
    elif ("Scaffold" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Scaffold ~~~~~~")
        Experiment_Scaffold(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                            testing_dataloader)

    # FedRep
    elif ("FedRep" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Rep ~~~~~~")
        Experiment_Federated_Rep(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader)
    # FedRep
    elif ("FedDC" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Daisy Chaining ~~~~~~")
        Experiment_Federated_Daisy_Chaining(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader)
    # FedSum
    elif ("FedSum" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Sum ~~~~~~")
        Experiment_Federated_Sum(param_dict, global_model, training_dataloaders, training_dataset,
                                            client_dataset_list, testing_dataloader)
    # FedSumNoProtoNoParam
    elif ("FedSumNoProtoNoParam" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Sum Without Proto and Param  ~~~~~~")
        Experiment_Federated_Sum_No_Proto_No_Param(param_dict, global_model, training_dataloaders, training_dataset,
                                            client_dataset_list, testing_dataloader)

    # FedProto
    elif ("FedProto" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Proto ~~~~~~")
        Experiment_Federated_Proto(param_dict, global_model, training_dataloaders, training_dataset,
                                 client_dataset_list, testing_dataloader)

    # Ditto
    elif ("Ditto" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Ditto ~~~~~~")
        Experiment_Ditto(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                         testing_dataloader)

    # Giant
    elif ("Giant".lower() in param_dict["algorithm"].lower()):
        logger.info("~~~~~~ Algorithm: Giant ~~~~~~")
        Experiment_Giant(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader
        )
    else:
        raise ValueError(f'''Wrong algorithm name:{param_dict['algorithm']} It should be in the following type:
            [Separate | FedAvg | FedProx | FedDC | FedSum | pFedMe | SuPerFed | Ditto | FedRep | Scaffold | Test | Giant] ''')
