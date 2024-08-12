from copy import deepcopy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig

from torch.nn.init import xavier_uniform_

from tool.logger import *
from hypothesis.TransformerEncoder import ExtTransformerEncoder

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if (large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            #self.model = BertModel.from_pretrained('longformer-base-4096', cache_dir=temp_dir)
            # self.model = AutoModel.from_pretrained('bert-base-uncased')
            # loaded_paras = torch.load("D:/06/temp/BERT/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.7156163d5fdc189c3016baca0775ffce230789d7fa2a42ef516483e4ca884517")
        #     loaded_paras = torch.load('D:/06/temp/BERT/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157')
        #
        #     state_dict = deepcopy(self.model.state_dict())
        #     loaded_paras_names = list(loaded_paras.keys())[:-8]
        #     model_paras_names = list(state_dict.keys())[1:]
        #     for i in range(len(loaded_paras_names)):
        #         # 判断当前参数是否为positional embeding层，如果是进行替换即可
        #         if "position_embeddings" in model_paras_names[i]:
        #             ### 这部分代码用来消除预训练模型只能输入小于512个字符的限制 begin ###
        #             if self.model.config.max_position_embeddings > 512:
        #                 self.replace_512_position(state_dict, loaded_paras[loaded_paras_names[i]])
        #             ### 这部分代码用来消除预训练模型只能输入小于512个字符的限制 begin ###
        #         else:
        #             state_dict[model_paras_names[i]] = loaded_paras[loaded_paras_names[i]]
        #         logging.debug(f"## 成功将参数:{loaded_paras_names[i]}赋值给{model_paras_names[i]},"
        #                       f"参数形状为:{state_dict[model_paras_names[i]].size()}")
        #     self.model.load_state_dict(state_dict)
        #
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if (self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec

    def replace_512_position(self, state_dict, loaded_embedding):
        logging.info("Model parameter max_positional_embedding > 512, using replacement!")
        state_dict['embeddings.position_embeddings.weight'][:512, :] = loaded_embedding[:512, :]


class ExtSummarizer(nn.Module):
    def __init__(self, classifier_type, forzen=False):
        super(ExtSummarizer, self).__init__()

        #为scaffold算法提供的控制变量
        self.control={}
        self.delta_control={}
        self.delta_y={}

        #为ditto算法创建的delta_global_model
        self.delta_global_model = {}

        # not pre-trained BERT and Linear head
        if 'baseline'.lower() in classifier_type.lower():
            self.bert = Bert(large=False, temp_dir="./temp/BERT", finetune=True)
            # bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=768, num_hidden_layers=2,
                                     # num_attention_heads=8, intermediate_size=2048)
            bert_config = BertConfig(30522, hidden_size=768, num_hidden_layers=2,
                                     num_attention_heads=8, intermediate_size=2048)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)
        # not pre-trained BERT and Transformer head
        elif 'random_transformer'.lower() in classifier_type.lower():
            self.bert = Bert(large=False, temp_dir="./temp/BERT", finetune=True)
            bert_config = BertConfig(30522, hidden_size=768, num_hidden_layers=6,
                                     num_attention_heads=8, intermediate_size=2048)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = ExtTransformerEncoder(768,  2048,  8, 0.2,  2)
            # self.ext_layer = Classifier(self.bert.model.config.hidden_size)
        # pre-trained BERT and Linear head
        elif 'linear'.lower() in classifier_type.lower():
            self.bert = Bert(False, "./temp/BERT", True)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)
        else:  # pre-trained BERT and Transformer head
            self.bert = Bert(False, "./temp/BERT", True)
            # bert_config = BertConfig(30522, hidden_size=768, num_hidden_layers=2,
            #                          num_attention_heads=8, intermediate_size=2048,max_position_embeddings = 1024)
            # self.bert.model = BertModel(bert_config)
            # for p in self.bert.parameters():
            #     p.requires_grad = False
            self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size,  2048,  8, 0.2,  2)


        for p in self.ext_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        # self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):

        top_vec = self.bert(src, segs, mask_src)
        # top_vec = top_vec.detach()
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        torch.cuda.empty_cache()
        return sent_scores, mask_cls

