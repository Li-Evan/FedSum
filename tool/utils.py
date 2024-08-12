#!/usr/bin/python
# -*- coding: utf-8 -*-
import gc
import re
import os
# import thop
import torch
import numpy as np
import scipy
import time
import argparse
import itertools
from bert_score import score
from rouge import Rouge
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List
from collections import OrderedDict
from tool.logger import *

sys.setrecursionlimit(10000)


def get_specific_time():
    now = time.localtime()
    year, month, day = str(now.tm_year), str(now.tm_mon), str(now.tm_mday)
    hour, minute, second = str(now.tm_hour), str(now.tm_min), str(now.tm_sec)
    return str(year + "_" + month + "_" + day + "_" + hour + "h" + minute + "m" + second + "s")


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    x = x.lower()
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def check_and_make_the_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# compute the cos similarity between a and b. a, b are numpy arrays
def cos_sim(a, b):
    return 1 - scipy.spatial.distance.cosine(a, b)


def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        print("match_true:", match_true.data, " ;pred:", pred.data, " ;true:", true.data, " ;match:", match.data,
              " ;total:", total)
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def get_tensor_parameters(net) -> List[torch.Tensor]:
    return list(net.parameters())


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def save_model(param_dict, updated_global_model, client_model_list, iter_t, optim):
    logger.info("Communication Round %d Global Models Saving" % (iter_t + 1))
    # TODO start
    model_state_dict = updated_global_model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        # 'generator': generator_state_dict,
        'opt': param_dict,
        'optims': optim,
    }
    check_and_make_the_path(param_dict['model_path'])
    torch.save(checkpoint, os.path.join(param_dict['model_path'], "step_%d_" % iter_t + "global_model.pt"))
    # TODO end
    # torch.save(updated_global_model, os.path.join(param_dict['model_path'], "step_%d_" % iter_t + "global_model.pkl"))
    logger.info("Communication Round %d Client Models Saving" % (iter_t + 1))
    for client_id, client_model in enumerate(client_model_list):
        _ = os.path.join(param_dict['model_path'],
                         "client_" + str(client_id + 1), "step_%d_" % iter_t + "model.pkl")
        check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
        torch.save(client_model, _)


def save_model_sepa(param_dict, client_model_list, epoch):
    check_and_make_the_path(param_dict['model_path'])
    logger.info("Total Epoch %d Separate Client Models Saving" % (epoch))
    for client_id, client_model in enumerate(client_model_list):
        _ = os.path.join(param_dict['model_path'],
                         "client_" + str(client_id + 1), "step_%d_" % epoch + "model.pkl")
        check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
        torch.save(client_model, _)


def get_train_sample():
    pass


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8', errors='ignore')]
    references = [line.strip() for line in open(ref, encoding='utf-8', errors='ignore')]
    # print(len(candidates))
    # print(len(references))
    # assert len(candidates) == len(references)

    pop_list = []
    for i in range(len(candidates)):
        candidate = candidates[i]
        if (len(candidate) == 1) or (candidate == "."):
            pop_list.append(i)
    for num, p in enumerate(pop_list):
        candidates.pop(p - num)
        references.pop(p - num)

    rouge = Rouge()
    results_dict = rouge.get_scores(candidates, references, avg=True, ignore_empty=True)  # a和b里面包含多个句子的时候用
    return results_dict


def test_2_gram(s1, s2):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    scores = scorer.score(s1, s2)
    return scores["rouge2"].recall


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\n>> ROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge-1"]["f"] * 100,
        results_dict["rouge-2"]["f"] * 100,
        results_dict["rouge-l"]["f"] * 100,
        results_dict["rouge-1"]["r"] * 100,
        results_dict["rouge-2"]["r"] * 100,
        results_dict["rouge-l"]["r"] * 100,

        ## 使用原版 rouge
        # results_dict["rouge_1_f_score"] * 100,
        # results_dict["rouge_2_f_score"] * 100,
        # # results_dict["rouge_3_f_score"] * 100,
        # results_dict["rouge_l_f_score"] * 100,
        # results_dict["rouge_1_recall"] * 100,
        # results_dict["rouge_2_recall"] * 100,
        # # results_dict["rouge_3_f_score"] * 100,
        # results_dict["rouge_l_recall"] * 100
        # ,results_dict["rouge_su*_f_score"] * 100
    )


def rouge_results_to_str_separate(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge-1-f"],
        results_dict["rouge-2-f"],
        results_dict["rouge-l-f"],
        results_dict["rouge-1-r"],
        results_dict["rouge-2-r"],
        results_dict["rouge-l-r"],

        ## 使用原版 rouge
        # results_dict["rouge_1_f_score"] * 100,
        # results_dict["rouge_2_f_score"] * 100,
        # # results_dict["rouge_3_f_score"] * 100,
        # results_dict["rouge_l_f_score"] * 100,
        # results_dict["rouge_1_recall"] * 100,
        # results_dict["rouge_2_recall"] * 100,
        # # results_dict["rouge_3_f_score"] * 100,
        # results_dict["rouge_l_recall"] * 100
        # ,results_dict["rouge_su*_f_score"] * 100
    )


# 巨人算法中途测试rouge
# sample_num_bar指的是每一次巨人算法中途计算rouge的时候要取多少个样本来计算（全部计算消耗可能太大了）
def test_rank(param_dict, device, testing_dataloader, testing_model_list, epoch, sample_num_bar=1):
    criterion = torch.nn.BCELoss(reduction='none')

    client_loss_score = [0 for _ in range(len(testing_model_list))]  # TODO:loss的值越大排名越靠前（排名的值越大）
    client_shortcut_score = [0 for _ in range(len(testing_model_list))]  # TODO:信息熵的值越小排名应该越前（排名的值越大）
    client_redundant_score = [0 for _ in range(len(testing_model_list))]  # TODO:句子冗余度越小排名越前

    # ## 随机抽取 sample_num_bar 份试卷大家一起考
    # # 使用random_split随机抽取k个元素
    # random_indices = torch.randperm(len(testing_dataloader.dataset))[:sample_num_bar]
    # random_dataset = torch.utils.data.Subset(testing_dataloader.dataset, random_indices)
    # # 创建新的dataloader
    # new_dataloader = DataLoader(dataset=random_dataset, batch_size=testing_dataloader.batch_size, shuffle=True)

    for id in range(len(testing_model_list)):
        torch.cuda.empty_cache()
        testing_model = testing_model_list[id]
        testing_model.eval()
        testing_model.zero_grad()
        testing_model.to(device)
        with torch.no_grad():
            for batch_index, batch in enumerate(testing_dataloader):
                torch.cuda.empty_cache()
                testing_model.zero_grad()
                src = batch['src'].to(device)
                labels = batch['src_sent_labels'].to(device)
                segs = batch['segs'].to(device)
                clss = batch['clss'].to(device)
                mask = batch['mask_src'].to(device)
                mask_cls = batch['mask_cls'].to(device)
                loss_list = []
                tmp_sent_scores_list = []
                for i in range(0, len(src), 4):
                    sbatch_size = src[i:i + 4].shape[0]  # 获取当前批次的样本数量
                    tmp_sent_scores, tmp_mask = testing_model(src[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                              segs[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                              clss[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                              mask[i:i + sbatch_size].reshape(sbatch_size, -1),
                                                              mask_cls[i:i + sbatch_size].reshape(sbatch_size, -1))
                    loss = criterion(tmp_sent_scores, labels[i:i + sbatch_size].reshape(sbatch_size, -1).float())
                    loss = (loss * tmp_mask.float()).sum()
                    loss = loss / sbatch_size
                    loss_list.append((loss / loss.numel()).data)
                    sent_scores = tmp_sent_scores + tmp_mask.float()
                    sent_scores = sent_scores.cpu().data.numpy()
                    tmp_sent_scores_list.append(sent_scores)
                epoch_loss = sum(loss_list) / len(loss_list)
                sent_scores_list = np.vstack(tmp_sent_scores_list)
                # sent_scores, mask = testing_model(src, segs, clss, mask, mask_cls)

                # 计算loss并保存
                # loss = criterion(sent_scores, labels.float())
                # loss = (loss * mask.float()).sum()
                client_loss_score[id] += epoch_loss.to("cpu")
                print(f"loss:{epoch_loss}")

                # 计算info_entropy并保存
                # sent_scores = sent_scores + mask.float()
                # sent_scores = sent_scores.cpu().data.numpy()
                # sent_scores = torch.from_numpy(sent_scores)
                sent_scores = torch.from_numpy(sent_scores_list)
                total_entropy = 0.0  # 用于存储总的信息熵
                # 循环遍历每个维度
                for dim in range(sent_scores.size(0)):
                    # 从张量中选择当前维度
                    current_dimension = sent_scores[dim]
                    # print(f"current_dimension:{current_dimension}")
                    # 计算信息熵
                    non_zero_elements = current_dimension[current_dimension != 0]
                    non_zero_elements = non_zero_elements - 1
                    # print(f"non_zero_elements:{non_zero_elements}")
                    total_probability = sum(non_zero_elements)
                    probabilities = torch.tensor([prob / total_probability if prob > 0 else 0.000001 for prob in
                                                  non_zero_elements])  # 计算非零元素的概率分布
                    # print(f"probabilities:{probabilities}")
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))  # 计算信息熵
                    # entropy = -torch.sum(non_zero_elements * torch.log2(non_zero_elements))  # 计算信息熵
                    # print(entropy)
                    total_entropy += entropy.item()  # 将信息熵添加到总信息熵中
                client_shortcut_score[id] += total_entropy
                print(f"total_entropy:{total_entropy}")

                # 计算冗余度并保存
                ext_num = 3
                if ("cnn" in param_dict["dataset".lower()]):
                    ext_num = 3
                elif ("xsum" in param_dict["dataset".lower()]):
                    ext_num = 2
                elif ("wiki" in param_dict["dataset".lower()]):
                    ext_num = 4
                elif ("pub" in param_dict["dataset".lower()]):
                    ext_num = 6
                elif ("government" in param_dict["dataset".lower()]):
                    ext_num = 7
                # selected_ids(i,j):i means the batch_size, j means sentence select possibility.so seleceted_ids[i][j] means a sentence
                selected_ids = np.argsort(-sent_scores, 1)
                # 里面是_pred，一个_pred对应一个样本，里面有ext_num句选择的句子
                pred = []
                for i, idx in enumerate(selected_ids):
                    if param_dict["dataset"] == "Mixtape":
                        ext_num = 3
                        if ("cnn" in batch["tag"][i].lower()):
                            ext_num = 3
                        elif ("xsum" in batch["tag"][i].lower()):
                            ext_num = 2
                        elif ("wiki" in batch["tag"][i].lower()):
                            ext_num = 4
                        elif ("pub" in batch["tag"][i].lower()):
                            ext_num = 6
                        elif ("government" in batch["tag"][i].lower()):
                            ext_num = 7
                    _pred = []
                    if (len(batch['src_txt'][i]) == 0):
                        continue
                    for j in selected_ids[i][:len(batch['src_txt'][i])]:
                        if (j >= len(batch['src_txt'][i])):
                            continue
                        candidate = batch['src_txt'][i][j].strip()
                        _pred.append(candidate)
                        if len(_pred) == ext_num:
                            break
                    pred.append(_pred)
                # 对pred里面的所有_pred，计算冗余度
                for p in pred:
                    combinations = list(itertools.combinations(p, 2))
                    # print(combinations)
                    num = len(combinations)
                    total_rouge2 = 0.0
                    for sen_part in combinations:
                        rouges = test_2_gram(sen_part[0], sen_part[1])
                        total_rouge2 += rouges * 100
                    if num > 1:
                        total_rouge2 /= num
                client_redundant_score[id] += total_rouge2
                print(f"redundency:{total_rouge2}")
                break
        del src, labels, segs, clss, mask, mask_cls
        testing_model.to("cpu")
        del testing_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    loss_ranked_list = np.argsort(client_loss_score)
    loss_ranked_list = [float(rk + 1) for rk in loss_ranked_list]
    print(f"shortcut:{client_shortcut_score}")
    import scipy.stats as stats
    shortcut_ranked_list = stats.rankdata(client_shortcut_score, method='ordinal')
    redundant_ranked_list = stats.rankdata(client_redundant_score, method='ordinal')
    return loss_ranked_list, shortcut_ranked_list, redundant_ranked_list


# TODO:衰减函数，在合成最终rank的时候要用到，传入两个基础rank，输出最终的rank
# 指数衰减，k为衰减速度参数
def attenuation_function(iter, k=0.5):
    return np.exp(-k * iter)


def test_bertscore(cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8', errors='ignore')]
    references = [line.strip() for line in open(ref, encoding='utf-8', errors='ignore')]

    P, R, F1 = score(candidates, references, lang="en", verbose=True)
    return f">> BertScore(P/R/F1):{P.mean():.3f}/{R.mean():.3f}/{F1.mean():.3f}\n"


def test_FLOPs(model, input_data):
    flops, params = thop.profile(model, inputs=input_data)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")


# 测试非sepa算法的rouge
def Testing_ROUGE(param_dict, device, testing_dataloader, testing_model, algorithm_epoch_T, communication_round_I):
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    testing_model.eval()
    testing_model.to(device)

    can_path = '%s/%s_tt_%s_%s_step%d_iter%d.candidate' % (
    "./results", param_dict['algorithm'], param_dict['tt'], param_dict["dataset_name"], algorithm_epoch_T,
    communication_round_I)
    gold_path = '%s/%s_tt_%s_%s_step%d_iter%d.gold' % (
    "./results", param_dict['algorithm'], param_dict['tt'], param_dict["dataset_name"], algorithm_epoch_T,
    communication_round_I)

    criterion = torch.nn.BCELoss(reduction='none')

    batch_loss_list = []
    algorithm_label_dis = torch.zeros(44).to(device)

    with open(can_path, 'w', encoding='utf-8') as save_pred:
        with open(gold_path, 'w', encoding='utf-8') as save_gold:
            with torch.no_grad():
                for batch_index, batch in enumerate(testing_dataloader):

                    # batch_size = batch['src'].size(0)  # 或者使用任何其他在batch中的张量
                    # print(f"Batch {batch_index} size: {batch_size}")

                    src = batch['src'].to(device)
                    labels = batch['src_sent_labels'].to(device)
                    segs = batch['segs'].to(device)
                    clss = batch['clss'].to(device)
                    mask = batch['mask_src'].to(device)
                    mask_cls = batch['mask_cls'].to(device)

                    gold = []
                    pred = []

                    sent_scores, mask = testing_model(src, segs, clss, mask, mask_cls)
                    # print(sent_scores.shape)

                    for row in range(sent_scores.shape[0]):  # 遍历32行
                        # 获取当前行
                        vector = sent_scores[row, :]
                        # 找出最大的3个值的索引
                        top_indices = torch.topk(vector, k=6).indices
                        # 在结果向量中将top3位置的值加1
                        algorithm_label_dis[top_indices] += 1

                    # print(algorithm_label_dis)
                    # Shape: [BatchSize, MaxLengthInTheBatch]
                    # 尺寸：【BatchSize，句子中最大词数目】
                    batch_loss = criterion(sent_scores, labels.float())
                    batch_loss = (batch_loss * mask.float()).sum()
                    batch_loss_list.append(batch_loss)

                    sent_scores = sent_scores + mask.float()
                    sent_scores = sent_scores.cpu().data.numpy()
                    selected_ids = np.argsort(-sent_scores, 1)

                    for i, idx in enumerate(selected_ids):
                        ext_num = 3
                        if ("cnn" in batch["tag"][i].lower()):
                            ext_num = 3
                        elif ("xsum" in batch["tag"][i].lower()):
                            ext_num = 2
                        elif ("wiki" in batch["tag"][i].lower()):
                            ext_num = 4
                        elif ("pub" in batch["tag"][i].lower()):
                            ext_num = 6
                        elif ("government" in batch["tag"][i].lower()):
                            ext_num = 7

                        _pred = []
                        if (len(batch['src_txt'][i]) == 0):
                            continue
                        for j in selected_ids[i][:len(batch['src_txt'][i])]:
                            if (j >= len(batch['src_txt'][i])):
                                continue
                            candidate = batch['src_txt'][i][j].strip()
                            if (not _block_tri(candidate, _pred)):
                                _pred.append(candidate)

                            if len(_pred) == ext_num:
                                break

                        _pred = '<q>'.join(_pred)

                        pred.append(_pred)
                        gold.append(batch['tgt_txt'][i])

                    empty_gold = []
                    for i in range(len(gold)):
                        if gold[i].strip() == "":
                            empty_gold.append(i)
                            # print(f"empty:{i}")
                        else:
                            save_gold.write(gold[i].strip() + '\n')

                    for i in range(len(pred)):
                        if i in empty_gold:
                            # print(f"emptypred:{i}")
                            continue
                        else:
                            save_pred.write(pred[i].strip() + '\n')

                    batch['src'].to("cpu")
                    batch['src_sent_labels'].to("cpu")
                    batch['segs'].to("cpu")
                    batch['clss'].to("cpu")
                    batch['mask_src'].to("cpu")
                    batch['mask_cls'].to("cpu")

                batch_avg_loss = sum(batch_loss_list) / len(batch_loss_list)
    print(algorithm_label_dis)
    distribution_sum = sum(algorithm_label_dis)
    distribution_ratio = algorithm_label_dis / distribution_sum
    distribution_ratio = np.array(distribution_ratio.tolist())
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    print(distribution_ratio)
    rouges = test_rouge("./temp", can_path, gold_path)
    logger.info(f'Rouges at ['
                f'dataset {param_dict["dataset"]};'
                f'algorithm {param_dict["algorithm"]};'
                f'split_strategy {param_dict["split_strategy"]};'
                f'client {param_dict["num_clients_K"]};'
                f'batch_size {param_dict["batch_size"]};'
                f'algorithm_epoch_T {algorithm_epoch_T};'
                f'communication_round_I {communication_round_I};'
                f'batch_avg_loss {batch_avg_loss};'
                f']\n' +
                rouge_results_to_str(rouges))

    testing_model.to("cpu")

    # bert_score = test_bertscore(can_path,gold_path)
    # logger.info(f'BertScore at ['
    #             f'dataset {param_dict["dataset"]};'
    #             f'algorithm {param_dict["algorithm"]};'
    #             f'split_strategy {param_dict["split_strategy"]};'
    #             f'client {param_dict["client"]};'
    #             f'batch_size {param_dict["batch_size"]};'
    #             f'algorithm_epoch_T {param_dict["algorithm_epoch_T"]};'
    #             f'communication_round_I {param_dict["communication_round_I"]};'
    #             f']\n' +
    #             bert_score)
    # logger.info(f'Rouges at ['
    #             'algorithm:%s '
    #             'epoch:%d '
    #             'communication:%d'
    #             '] \n%s'
    #             % (param_dict["algorithm"] ,epoch,iter, rouge_results_to_str(rouges)))

    # logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))


# 测试sepa算法的rouge
def Testing_ROUGE_separate(param_dict, device, testing_dataloader, testing_model_list, epoch):
    # handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    #
    # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # print("测试前GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    results_dict = {}
    results_dict["rouge-1-f"] = 0
    results_dict["rouge-2-f"] = 0
    results_dict["rouge-l-f"] = 0
    results_dict["rouge-1-r"] = 0
    results_dict["rouge-2-r"] = 0
    results_dict["rouge-l-r"] = 0

    client_rouge_list = list()

    for id in range(len(testing_model_list)):
        torch.cuda.empty_cache()
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("加载前GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
        testing_model = testing_model_list[id]
        testing_model.eval()
        # 加上的
        testing_model.zero_grad()
        testing_model.to(device)
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("加载后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
        step = 0
        can_path = '%s_step%d.candidate' % ("./results/%s" % param_dict["dataset_name"], epoch)
        gold_path = '%s_step%d.gold' % ("./results/%s" % param_dict["dataset_name"], epoch)
        # criterion = torch.nn.BCELoss(reduction='none')
        # logger.info("can_path:" + can_path)
        # logger.info("gold_path:" + gold_path)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch_index, batch in enumerate(testing_dataloader):
                        torch.cuda.empty_cache()
                        testing_model.zero_grad()
                        src = batch['src'].to(device)
                        segs = batch['segs'].to(device)
                        clss = batch['clss'].to(device)
                        mask = batch['mask_src'].to(device)
                        mask_cls = batch['mask_cls'].to(device)

                        gold = []
                        pred = []

                        sent_scores, mask = testing_model(src, segs, clss, mask, mask_cls)
                        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # print("测试后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
                        # 没有用到
                        # loss = criterion(sent_scores, labels.float())
                        # loss = (loss * mask.float()).sum()

                        sent_scores = sent_scores + mask.float()
                        sent_scores = sent_scores.cpu().data.numpy()
                        selected_ids = np.argsort(-sent_scores, 1)

                        # 推理完一个模型清空显存
                        # del src, labels, segs, clss, mask, mask_cls
                        # gc.collect()
                        # torch.cuda.empty_cache()
                        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # print("删除后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
                        # 删掉模型
                        # del testing_model

                        for i, idx in enumerate(selected_ids):
                            ext_num = 3
                            if ("cnn" in batch["tag"][i].lower()):
                                ext_num = 3
                            elif ("xsum" in batch["tag"][i].lower()):
                                ext_num = 2
                            elif ("wiki" in batch["tag"][i].lower()):
                                ext_num = 4
                            elif ("pub" in batch["tag"][i].lower()):
                                ext_num = 6
                            elif ("government" in batch["tag"][i].lower()):
                                ext_num = 7

                            _pred = []
                            if (len(batch['src_txt'][i]) == 0):
                                continue
                            for j in selected_ids[i][:len(batch['src_txt'][i])]:
                                if (j >= len(batch['src_txt'][i])):
                                    continue
                                candidate = batch['src_txt'][i][j].strip()
                                if (not _block_tri(candidate, _pred)):
                                    _pred.append(candidate)

                                if len(_pred) == ext_num:
                                    break

                            _pred = '<q>'.join(_pred)

                            pred.append(_pred)
                            gold.append(batch['tgt_txt'][i])

                        empty_gold = []
                        for i in range(len(gold)):
                            if gold[i].strip() == "":
                                empty_gold.append(i)
                                # print(f"empty:{i}")
                            else:
                                save_gold.write(gold[i].strip() + '\n')

                        for i in range(len(pred)):
                            if i in empty_gold:
                                # print(f"emptypred:{i}")
                                continue
                            else:
                                save_pred.write(pred[i].strip() + '\n')

                        batch['src'].to("cpu")
                        batch['segs'].to("cpu")
                        batch['clss'].to("cpu")
                        batch['mask_src'].to("cpu")
                        batch['mask_cls'].to("cpu")

        # 删掉模型
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("删除前GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
        del src, segs, clss, mask, mask_cls
        testing_model.to("cpu")
        del testing_model
        torch.cuda.empty_cache()
        gc.collect()
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("删除后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))

        rouges = test_rouge("./temp", can_path, gold_path)
        logger.info(f"id:{id}")
        logger.info(f"rouges:" + rouge_results_to_str(rouges))

        results_dict["rouge-1-f"] += rouges["rouge-1"]["f"] * 100
        results_dict["rouge-2-f"] += rouges["rouge-2"]["f"] * 100
        results_dict["rouge-l-f"] += rouges["rouge-l"]["f"] * 100
        results_dict["rouge-1-r"] += rouges["rouge-1"]["r"] * 100
        results_dict["rouge-2-r"] += rouges["rouge-2"]["r"] * 100
        results_dict["rouge-l-r"] += rouges["rouge-l"]["r"] * 100
        logger.info(f"len of testing model list:{len(testing_model_list)}")

        # temp_dict记录一个client的rouge值
        temp_dict = {}
        temp_dict["rouge-1-f"] = rouges["rouge-1"]["f"] * 100
        temp_dict["rouge-2-f"] = rouges["rouge-2"]["f"] * 100
        temp_dict["rouge-l-f"] = rouges["rouge-l"]["f"] * 100
        temp_dict["rouge-1-r"] = rouges["rouge-1"]["r"] * 100
        temp_dict["rouge-2-r"] = rouges["rouge-2"]["r"] * 100
        temp_dict["rouge-l-r"] = rouges["rouge-l"]["r"] * 100
        client_rouge_list.append(temp_dict)

    rouge_1_f = list()
    rouge_2_f = list()
    rouge_l_f = list()
    rouge_1_r = list()
    rouge_2_r = list()
    rouge_l_r = list()

    for dic in client_rouge_list:
        rouge_1_f.append(dic["rouge-1-f"])
        rouge_2_f.append(dic["rouge-2-f"])
        rouge_l_f.append(dic["rouge-l-f"])
        rouge_1_r.append(dic["rouge-1-r"])
        rouge_2_r.append(dic["rouge-2-r"])
        rouge_l_r.append(dic["rouge-l-r"])

    rouge_1_f_mean = np.mean(rouge_1_f)
    rouge_2_f_mean = np.mean(rouge_2_f)
    rouge_l_f_mean = np.mean(rouge_l_f)
    rouge_1_r_mean = np.mean(rouge_1_r)
    rouge_2_r_mean = np.mean(rouge_2_r)
    rouge_l_r_mean = np.mean(rouge_l_r)
    logger.info(f'rouge-1-f mean {rouge_1_f_mean}\n'
                f'rouge-2-f mean {rouge_2_f_mean}\n'
                f'rouge-l-f mean {rouge_l_f_mean}\n'
                f'rouge-1-r mean {rouge_1_r_mean}\n'
                f'rouge-2-r mean {rouge_2_r_mean}\n'
                f'rouge-l-r mean {rouge_l_r_mean}\n')

    rouge_1_f_var = np.var(rouge_1_f)
    rouge_2_f_var = np.var(rouge_2_f)
    rouge_l_f_var = np.var(rouge_l_f)
    rouge_1_r_var = np.var(rouge_1_r)
    rouge_2_r_var = np.var(rouge_2_r)
    rouge_l_r_var = np.var(rouge_l_r)

    logger.info(f'rouge-1-f variance {rouge_1_f_var}\n'
                f'rouge-2-f variance {rouge_2_f_var}\n'
                f'rouge-l-f variance {rouge_l_f_var}\n'
                f'rouge-1-r variance {rouge_1_r_var}\n'
                f'rouge-2-r variance {rouge_2_r_var}\n'
                f'rouge-l-r variance {rouge_l_r_var}\n')

    results_dict["rouge-1-f"] = results_dict["rouge-1-f"] / len(testing_model_list)
    results_dict["rouge-2-f"] = results_dict["rouge-2-f"] / len(testing_model_list)
    results_dict["rouge-l-f"] = results_dict["rouge-l-f"] / len(testing_model_list)
    results_dict["rouge-1-r"] = results_dict["rouge-1-r"] / len(testing_model_list)
    results_dict["rouge-2-r"] = results_dict["rouge-2-r"] / len(testing_model_list)
    results_dict["rouge-l-r"] = results_dict["rouge-l-r"] / len(testing_model_list)
    logger.info(f'Rouges at ['
                f'dataset {param_dict["dataset"]};'
                f'algorithm {param_dict["algorithm"]};'
                f'split_strategy {param_dict["split_strategy"]};'
                f'client {param_dict["num_clients_K"]};'
                f'batch_size {param_dict["batch_size"]};'
                f'algorithm_epoch_T {param_dict["algorithm_epoch_T"]};'
                f'communication_round_I {param_dict["communication_round_I"]};'
                f']\n' +
                rouge_results_to_str(rouges))


# 测试个性化FL算法的rouge
def Testing_ROUGE_Personalize(param_dict, device, testing_dataloader, testing_model_list, algorithm_epoch_T,
                              communication_round_I):
    # handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    #
    # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # print("测试前GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    results_dict = {}
    results_dict["rouge-1-f"] = 0
    results_dict["rouge-2-f"] = 0
    results_dict["rouge-l-f"] = 0
    results_dict["rouge-1-r"] = 0
    results_dict["rouge-2-r"] = 0
    results_dict["rouge-l-r"] = 0

    client_rouge_list = list()

    for id in range(len(testing_model_list)):
        torch.cuda.empty_cache()
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("加载前GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
        testing_model = testing_model_list[id]
        testing_model.eval()
        # 加上的
        testing_model.zero_grad()
        testing_model.to(device)
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("加载后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))

        can_path = '%s/%s_tt_%s_%s_client_%d_step%d_iter%d.candidate' % (
            "./results", param_dict['algorithm'], param_dict['tt'], param_dict["dataset_name"], id, algorithm_epoch_T,
            communication_round_I)
        gold_path = '%s/%s_tt_%s_%s_client_%d_step%d_iter%d.gold' % (
            "./results", param_dict['algorithm'], param_dict['tt'], param_dict["dataset_name"], id, algorithm_epoch_T,
            communication_round_I)

        # criterion = torch.nn.BCELoss(reduction='none')
        logger.info("can_path:" + can_path)
        logger.info("gold_path:" + gold_path)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch_index, batch in enumerate(testing_dataloader):
                        torch.cuda.empty_cache()
                        testing_model.zero_grad()
                        src = batch['src'].to(device)
                        segs = batch['segs'].to(device)
                        clss = batch['clss'].to(device)
                        mask = batch['mask_src'].to(device)
                        mask_cls = batch['mask_cls'].to(device)

                        gold = []
                        pred = []

                        sent_scores, mask = testing_model(src, segs, clss, mask, mask_cls)
                        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # print("测试后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
                        # 没有用到
                        # loss = criterion(sent_scores, labels.float())
                        # loss = (loss * mask.float()).sum()

                        sent_scores = sent_scores + mask.float()
                        sent_scores = sent_scores.cpu().data.numpy()
                        selected_ids = np.argsort(-sent_scores, 1)

                        # 推理完一个模型清空显存
                        # del src, labels, segs, clss, mask, mask_cls
                        # gc.collect()
                        # torch.cuda.empty_cache()
                        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # print("删除后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
                        # 删掉模型
                        # del testing_model

                        for i, idx in enumerate(selected_ids):
                            ext_num = 3
                            if ("cnn" in batch["tag"][i].lower()):
                                ext_num = 3
                            elif ("xsum" in batch["tag"][i].lower()):
                                ext_num = 2
                            elif ("wiki" in batch["tag"][i].lower()):
                                ext_num = 4
                            elif ("pub" in batch["tag"][i].lower()):
                                ext_num = 6
                            elif ("government" in batch["tag"][i].lower()):
                                ext_num = 7

                            _pred = []
                            if (len(batch['src_txt'][i]) == 0):
                                continue
                            for j in selected_ids[i][:len(batch['src_txt'][i])]:
                                if (j >= len(batch['src_txt'][i])):
                                    continue
                                candidate = batch['src_txt'][i][j].strip()
                                if (not _block_tri(candidate, _pred)):
                                    _pred.append(candidate)

                                if len(_pred) == ext_num:
                                    break

                            _pred = '<q>'.join(_pred)

                            pred.append(_pred)
                            gold.append(batch['tgt_txt'][i])

                        empty_gold = []
                        for i in range(len(gold)):
                            if gold[i].strip() == "":
                                empty_gold.append(i)
                                # print(f"empty:{i}")
                            else:
                                save_gold.write(gold[i].strip() + '\n')

                        for i in range(len(pred)):
                            if i in empty_gold:
                                # print(f"emptypred:{i}")
                                continue
                            else:
                                save_pred.write(pred[i].strip() + '\n')

                        batch['src'].to("cpu")
                        batch['segs'].to("cpu")
                        batch['clss'].to("cpu")
                        batch['mask_src'].to("cpu")
                        batch['mask_cls'].to("cpu")

        # 删掉模型
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("删除前GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))
        del src, segs, clss, mask, mask_cls
        testing_model.to("cpu")
        del testing_model
        torch.cuda.empty_cache()
        gc.collect()
        # memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("删除后GPU %d Memory Used: %.4f G"%(1,memo_info.used/1024/1024/1000))

        rouges = test_rouge("./temp", can_path, gold_path)
        logger.info(f"id:{id}")
        logger.info(f"rouges:" + rouge_results_to_str(rouges))

        results_dict["rouge-1-f"] += rouges["rouge-1"]["f"] * 100
        results_dict["rouge-2-f"] += rouges["rouge-2"]["f"] * 100
        results_dict["rouge-l-f"] += rouges["rouge-l"]["f"] * 100
        results_dict["rouge-1-r"] += rouges["rouge-1"]["r"] * 100
        results_dict["rouge-2-r"] += rouges["rouge-2"]["r"] * 100
        results_dict["rouge-l-r"] += rouges["rouge-l"]["r"] * 100
        logger.info(f"len of testing model list:{len(testing_model_list)}")

        # temp_dict记录一个client的rouge值
        temp_dict = {}
        temp_dict["rouge-1-f"] = rouges["rouge-1"]["f"] * 100
        temp_dict["rouge-2-f"] = rouges["rouge-2"]["f"] * 100
        temp_dict["rouge-l-f"] = rouges["rouge-l"]["f"] * 100
        temp_dict["rouge-1-r"] = rouges["rouge-1"]["r"] * 100
        temp_dict["rouge-2-r"] = rouges["rouge-2"]["r"] * 100
        temp_dict["rouge-l-r"] = rouges["rouge-l"]["r"] * 100
        client_rouge_list.append(temp_dict)

    rouge_1_f = list()
    rouge_2_f = list()
    rouge_l_f = list()
    rouge_1_r = list()
    rouge_2_r = list()
    rouge_l_r = list()

    for dic in client_rouge_list:
        rouge_1_f.append(dic["rouge-1-f"])
        rouge_2_f.append(dic["rouge-2-f"])
        rouge_l_f.append(dic["rouge-l-f"])
        rouge_1_r.append(dic["rouge-1-r"])
        rouge_2_r.append(dic["rouge-2-r"])
        rouge_l_r.append(dic["rouge-l-r"])

    rouge_1_f_mean = np.mean(rouge_1_f)
    rouge_2_f_mean = np.mean(rouge_2_f)
    rouge_l_f_mean = np.mean(rouge_l_f)
    rouge_1_r_mean = np.mean(rouge_1_r)
    rouge_2_r_mean = np.mean(rouge_2_r)
    rouge_l_r_mean = np.mean(rouge_l_r)
    logger.info(f'rouge-1-f mean {rouge_1_f_mean}\n'
                f'rouge-2-f mean {rouge_2_f_mean}\n'
                f'rouge-l-f mean {rouge_l_f_mean}\n'
                f'rouge-1-r mean {rouge_1_r_mean}\n'
                f'rouge-2-r mean {rouge_2_r_mean}\n'
                f'rouge-l-r mean {rouge_l_r_mean}\n')

    rouge_1_f_var = np.var(rouge_1_f)
    rouge_2_f_var = np.var(rouge_2_f)
    rouge_l_f_var = np.var(rouge_l_f)
    rouge_1_r_var = np.var(rouge_1_r)
    rouge_2_r_var = np.var(rouge_2_r)
    rouge_l_r_var = np.var(rouge_l_r)

    logger.info(f'rouge-1-f variance {rouge_1_f_var}\n'
                f'rouge-2-f variance {rouge_2_f_var}\n'
                f'rouge-l-f variance {rouge_l_f_var}\n'
                f'rouge-1-r variance {rouge_1_r_var}\n'
                f'rouge-2-r variance {rouge_2_r_var}\n'
                f'rouge-l-r variance {rouge_l_r_var}\n')

    results_dict["rouge-1-f"] = results_dict["rouge-1-f"] / len(testing_model_list)
    results_dict["rouge-2-f"] = results_dict["rouge-2-f"] / len(testing_model_list)
    results_dict["rouge-l-f"] = results_dict["rouge-l-f"] / len(testing_model_list)
    results_dict["rouge-1-r"] = results_dict["rouge-1-r"] / len(testing_model_list)
    results_dict["rouge-2-r"] = results_dict["rouge-2-r"] / len(testing_model_list)
    results_dict["rouge-l-r"] = results_dict["rouge-l-r"] / len(testing_model_list)
    logger.info(f'Rouges at ['
                f'dataset {param_dict["dataset"]};'
                f'algorithm {param_dict["algorithm"]};'
                f'split_strategy {param_dict["split_strategy"]};'
                f'client {param_dict["num_clients_K"]};'
                f'batch_size {param_dict["batch_size"]};'
                f'algorithm_epoch_T {param_dict["algorithm_epoch_T"]};'
                f'communication_round_I {param_dict["communication_round_I"]};'
                f']\n' +
                rouge_results_to_str(rouges))
