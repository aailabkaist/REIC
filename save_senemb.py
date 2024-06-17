import json
import random
from functools import partial

import numpy as np
import redis
import sklearn
import torch
from eveliver import (Logger, Trainer, TrainerCallback, load_model,
                      tensor_to_obj)
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm
import logging


def eval_performance(facts, pred_result):
    sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec = []
    rec = []
    correct = 0
    total = len(facts)
    for i, item in enumerate(sorted_pred_result):
        if (item['entpair'][0], item['entpair'][1], item['relation']) in facts:
            correct += 1
        prec.append(float(correct) / float(i + 1))
        rec.append(float(correct) / float(total))
    auc = sklearn.metrics.auc(x=rec, y=prec)
    np_prec = np.array(prec)
    np_rec = np.array(rec)
    f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    mean_prec = np_prec.mean()
    return {'prec': np_prec.tolist(), 'rec': np_rec.tolist(), 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}


def expand(start, end, total_len, max_size):
    e_size = max_size - (end - start)
    _1 = start - (e_size // 2)
    _2 = end + (e_size - e_size // 2)
    if _2 - _1 <= total_len:
        if _1 < 0:
            _2 -= -1
            _1 = 0
        elif _2 > total_len:
            _1 -= (_2 - total_len)
            _2 = total_len
    else:
        _1 = 0
        _2 = total_len
    return _1, _2


def place_train_data(dataset):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, 'n/a', l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            for label in labels:
                if label != 'n/a':
                    ds = l2docs[label]
                    if 'n/a' in l2docs:
                        ds.extend(l2docs['n/a'])
                    bags.append([key, label, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + x[1])
    return bags


def place_dev_data(dataset, single_path):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, ['n/a'], l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            ds = list()
            for label in labels:
                if single_path and label != 'n/a':
                    ds.append(random.choice(l2docs[label]))
                else:
                    ds.extend(l2docs[label])
            if 'n/a' in labels:
                labels.remove('n/a')
            bags.append([key, labels, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + '#'.join(x[1]))
    return bags


def gen_c(tokenizer, passage, span, max_len, bound_tokens, d_start, d_end, no_additional_marker, mask_entity, sent_model):
    ret = list()
    ret.append(bound_tokens[0])
    for i in range(span[0], span[1]):
        if mask_entity:
            ret.append('[MASK]')
        else:
            ret.append(passage[i])
    ret.append(bound_tokens[1])
    prev = list()
    prev_ptr = span[0] - 1
    # while len(prev) < max_len:
    while True:
        if prev_ptr < 0:
            break
        if not no_additional_marker and prev_ptr in d_end:
            prev.append(f'[unused{(d_end[prev_ptr] + 2) * 2 + 2}]')
        prev.append(passage[prev_ptr])
        if not no_additional_marker and prev_ptr in d_start:
            prev.append(f'[unused{(d_start[prev_ptr] + 2) * 2 + 1}]')
        prev_ptr -= 1
    nex = list()
    nex_ptr = span[1]
    # while len(nex) < max_len:
    while True:
        if nex_ptr >= len(passage):
            break
        if not no_additional_marker and nex_ptr in d_start:
            nex.append(f'[unused{(d_start[nex_ptr] + 2) * 2 + 1}]')
        nex.append(passage[nex_ptr])
        if not no_additional_marker and nex_ptr in d_end:
            nex.append(f'[unused{(d_end[nex_ptr] + 2) * 2 + 2}]')
        nex_ptr += 1
    prev.reverse()
    ret = prev + ret + nex

    BLOCK_SIZE = 63
    cnt = 0
    ret0 = list()
    BERT_MAX_LEN = 512

    end_tokens = {'\n': 0, '.': 1, '?': 1, '!': 1, ',': 2}
    for k, v in list(end_tokens.items()):
        end_tokens['Ġ' + k] = v
    sen_cost, break_cost = 4, 8
    poses = [(i, end_tokens[tok]) for i, tok in enumerate(ret) if tok in end_tokens]
    poses.insert(0, (-1, 0))
    if poses[-1][0] < len(ret) - 1:
        poses.append((len(ret) - 1, 0))
    x = 0
    while x < len(poses) - 1:
        if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
            poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
        x += 1

    best = [(0, 0)]
    for i, (p, cost) in enumerate(poses):
        if i == 0:
            continue
        best.append((-1, 100000))
        for j in range(i - 1, -1, -1):
            if p - poses[j][0] > BLOCK_SIZE:
                break
            value = best[j][1] + cost + sen_cost
            if value < best[i][1]:
                best[i] = (j, value)
        assert best[i][0] >= 0
    intervals, x = [], len(poses) - 1
    while x > 0:
        l = poses[best[x][0]][0]
        intervals.append((l + 1, poses[x][0] + 1))
        x = best[x][0]

    target_start_index = ret.index(bound_tokens[0])
    target_end_index = ret.index(bound_tokens[1])

    intervals = intervals[::-1]

    target_sent_index = -1
    for idx, (st, en) in enumerate(intervals):
        if st <= target_start_index < en:
            target_sent_index = idx
            target_sent_start = st
            start_sent_end = en
            intervals.remove((st, en))
    if target_sent_index == -1:
        raise NotImplementedError(f"target_start_index: {target_start_index}, intervals: {intervals}")

    target_tokens = ret[target_sent_start:start_sent_end]
    try:
        target_tokens.remove(bound_tokens[0])
    except:
        print(f'no head token: {target_tokens}')
    try:
        target_tokens.remove(bound_tokens[1])
    except:
        print(f'no tail token: {target_tokens}')
    target_tokens = ['[CLS]'] + target_tokens
    target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    if len(target_token_ids) < BERT_MAX_LEN:
        target_token_ids = target_token_ids + [0] * (BERT_MAX_LEN - len(target_token_ids))
    target_attention_mask = [1] * len(target_tokens) + [0] * (BERT_MAX_LEN - len(target_tokens))
    target_token_ids = torch.tensor(target_token_ids, dtype=torch.int64).unsqueeze(0).cuda()
    target_attention_mask = torch.tensor(target_attention_mask, dtype=torch.int64).unsqueeze(0).cuda()
    target_emb = sent_model(target_token_ids, target_attention_mask)[0][:, 0].data.cpu().numpy().tolist()

    lst_emb = []
    for idx, (st, en) in enumerate(intervals):
        tmp_tokens = ret[st:en]
        if idx < target_sent_index:
            tmp_tokens = ['[CLS]'] + tmp_tokens + ['[SEP]'] + target_tokens
        else:
            tmp_tokens = ['[CLS]'] + target_tokens + ['[SEP]'] + tmp_tokens
        tmp_token_ids = tokenizer.convert_tokens_to_ids(tmp_tokens)
        tmp_token_ids = [x for x in tmp_token_ids if x >= 100]
        if len(tmp_token_ids) < BERT_MAX_LEN:
            tmp_token_ids = tmp_token_ids + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
        tmp_attention_mask = [1] * len(tmp_token_ids) + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
        tmp_token_ids = torch.tensor(tmp_token_ids, dtype=torch.int64).unsqueeze(0).cuda()
        tmp_attention_mask = torch.tensor(tmp_attention_mask, dtype=torch.int64).unsqueeze(0).cuda()
        tmp_emb = sent_model(tmp_token_ids, tmp_attention_mask)[0][:, 0].data.cpu().numpy().tolist()

        lst_emb.append(tmp_emb)

    return ret, intervals, target_emb, lst_emb


def process_example(args, h, t, doc1, doc2, tokenizer, max_len, redisd, no_additional_marker, mask_entity, sent_model):
    doc1_id = doc1
    doc2_id = doc2

    doc1 = json.loads(redisd.get('codred-doc-' + doc1_id))
    doc2 = json.loads(redisd.get('codred-doc-' + doc2_id))

    v_h = None
    for entity in doc1['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == h and v_h is None:
            v_h = entity
    assert v_h is not None
    v_t = None
    for entity in doc2['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == t and v_t is None:
            v_t = entity
    assert v_t is not None
    d1_v = dict()
    for entity in doc1['entities']:
        if 'Q' in entity:
            d1_v[entity['Q']] = entity
    d2_v = dict()
    for entity in doc2['entities']:
        if 'Q' in entity:
            d2_v[entity['Q']] = entity
    ov = set(d1_v.keys()) & set(d2_v.keys())
    if len(ov) > 40:
        ov = set(random.choices(list(ov), k=40))
    ov = list(ov)
    ma = dict()
    for e in ov:
        ma[e] = len(ma)
    d1_start = dict()
    d1_end = dict()
    for entity in doc1['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d1_start[span[0]] = ma[entity['Q']]
                d1_end[span[1] - 1] = ma[entity['Q']]
    d2_start = dict()
    d2_end = dict()
    for entity in doc2['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d2_start[span[0]] = ma[entity['Q']]
                d2_end[span[1] - 1] = ma[entity['Q']]
    k1, intervals1, target_emb1, lst_emb1 = gen_c(tokenizer, doc1['tokens'], v_h['spans'][0], max_len // 2 - 2, ['[unused1]', '[unused2]'], d1_start, d1_end, no_additional_marker, mask_entity, sent_model)
    k2, intervals2, target_emb2, lst_emb2 = gen_c(tokenizer, doc2['tokens'], v_t['spans'][0], max_len // 2 - 1, ['[unused3]', '[unused4]'], d2_start, d2_end, no_additional_marker, mask_entity, sent_model)

    senemb1 = {
        'tokens': k1,
        'intervals': intervals1,
        'target_emb': target_emb1,
        'lst_emb': lst_emb1
    }

    senemb2 = {
        'tokens': k2,
        'intervals': intervals2,
        'target_emb': target_emb2,
        'lst_emb': lst_emb2
    }

    senemb_path = args.senemb_path

    with open(f'{senemb_path}/train/codred-doc-senemb-doc1-{h}-{t}-{doc1_id}-{doc2_id}', "w") as json_file:
        json.dump(senemb1, json_file)
    with open(f'{senemb_path}/train/codred-doc-senemb-doc2-{h}-{t}-{doc1_id}-{doc2_id}', "w") as json_file:
        json.dump(senemb2, json_file)

    # redisd.set(f'codred-doc-senemb-{doc1_id}', json.dumps(senemb1))
    # redisd.set(f'codred-doc-senemb-{doc2_id}', json.dumps(senemb2))

    '''
    tokens = ['[CLS]'] + k1 + ['[SEP]'] + k2 + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) < max_len:
        token_ids = token_ids + [0] * (max_len - len(tokens))
    attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    token_type_id = [0] * (len(k1) + 2) + [1] * (len(k2) + 1) + [0] * (max_len - len(tokens))
    return tokens, token_ids, token_type_id, attention_mask
    '''
    return k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2


def collate_fn(lst_batch, args, relation2id, tokenizer, redisd, sent_model):
    # assert len(batch) == 1
    lst_input_ids_t = []
    lst_token_type_ids_t = []
    lst_attention_mask_t = []
    lst_dplabel_t = []
    lst_rs_t = []
    lst_r = []


    lst_tokens_codre = []
    lst_intervals_codre = []
    lst_target_emb_codre = []
    lst_lst_emb_codre = []
    lst_dp_label_codre = []
    lst_rs_codre = []
    lst_r_codre = []

    for i in range(len(lst_batch)):
        batch = lst_batch[i]
        if batch[-1] == 'o':
            h, t = batch[0].split('#')
            r = relation2id[batch[1]]
            dps = batch[2]
            if len(dps) > 8:
                dps = random.choices(dps, k=8)
            tokens_codre = list()
            intervals_codre = list()
            target_emb_codre = list()
            lst_emb_codre = list()
            dplabel = list()
            for doc1, doc2, l in dps:
                # tokens, token_ids, token_type_id, amask = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
                k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2 = process_example(args, h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
                continue
        else:
            continue

    return lst_input_ids_t, lst_token_type_ids_t, lst_attention_mask_t, lst_dplabel_t, lst_rs_t, lst_r,\
           lst_tokens_codre, lst_intervals_codre, lst_target_emb_codre, lst_lst_emb_codre, lst_dp_label_codre, lst_rs_codre, lst_r_codre


def collate_fn_infer(batch, args, relation2id, tokenizer, redisd, sent_model):
    assert len(batch) == 1
    batch = batch[0]
    h, t = batch[0].split('#')
    rs = [relation2id[r] for r in batch[1]]
    dps = batch[2]
    input_ids = list()
    token_type_ids = list()
    attention_mask = list()
    for doc1, doc2, l in dps:
        tokens, token_ids, token_type_id, amask = process_example(args, h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
        input_ids.append(token_ids)
        token_type_ids.append(token_type_id)
        attention_mask.append(amask)
    input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
    token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
    attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)

    lst_input_ids_t = torch.stack([input_ids_t])
    lst_token_type_ids_t = torch.stack([token_type_ids_t])
    lst_attention_mask_t = torch.stack([attention_mask_t])

    return lst_input_ids_t, lst_token_type_ids_t, lst_attention_mask_t, h, rs, t


class Codred(torch.nn.Module):
    def __init__(self, args, num_relations):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.predictor = torch.nn.Linear(self.bert.config.hidden_size, num_relations)
        weight = torch.ones(num_relations, dtype=torch.float32)
        weight[0] = 0.1
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weight)
        self.aggregator = args.aggregator
        self.no_doc_pair_supervision = args.no_doc_pair_supervision
    
    def forward(self, lst_input_ids, lst_token_type_ids, lst_attention_mask, lst_dplabel=None, lst_rs=None):
        f_loss = 0.
        f_prediction = []
        f_logit = []
        for i in range(len(lst_input_ids)):
            input_ids = lst_input_ids[i].to(self.bert.device)
            token_type_ids = lst_token_type_ids[i].to(self.bert.device)
            attention_mask = lst_attention_mask[i].to(self.bert.device)
            if lst_dplabel is not None:
                dplabel = lst_dplabel[i]
            else:
                dplabel = None
            if lst_rs is not None:
                rs = lst_rs[i]
            else:
                rs = None
            # input_ids: T(num_sentences, seq_len)
            # token_type_ids: T(num_sentences, seq_len)
            # attention_mask: T(num_sentences, seq_len)
            # rs: T(batch_size)
            # maps: T(batch_size, max_bag_size)
            # embedding: T(num_sentences, seq_len, embedding_size)
            embedding, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
            # r_embedding: T(num_sentences, embedding_size)
            r_embedding = embedding[:, 0, :]
            # logit: T(1, num_relations)
            # dp_logit: T(num_sentences, num_relations)
            logit, dp_logit = self.predict_logit(r_embedding, rs=rs)
            # prediction: T(1)
            _, prediction = torch.max(logit, dim=1)
            if dplabel is not None and rs is None:
                loss = self.loss(dp_logit, dplabel.to(self.bert.device))
                # prediction: T(num_sentences)
                _, prediction = torch.max(dp_logit, dim=1)
            elif rs is not None:
                if self.no_doc_pair_supervision:
                    loss = self.loss(logit, rs.to(self.bert.device))
                else:
                    loss = self.loss(logit, rs.to(self.bert.device)) + self.loss(dp_logit, dplabel.to(self.bert.device))
            else:
                loss = None
            if loss is not None:
                f_loss += loss
            f_prediction.append(prediction)
            f_logit.append(logit)
        f_loss /= len(lst_input_ids)
        return f_loss, f_prediction, f_logit
    
    def predict_logit(self, r_embedding, rs=None):
        # r_embedding: T(num_sentences, embedding_size)
        # weight: T(num_relations, embedding_size)
        weight = self.predictor.weight
        if self.aggregator == 'max':
            # scores: T(num_sentences, num_relations)
            scores = self.predictor(r_embedding)
            # prob: T(num_sentences, num_relations)
            prob = torch.nn.functional.softmax(scores, dim=1)
            if rs is not None:
                _, idx = torch.max(prob[:, rs[0]], dim=0, keepdim=True)
                return scores[idx], scores
            else:
                # max_score: T(1, num_relations)
                max_score, _ = torch.max(scores, dim=0, keepdim=True)
                return max_score, scores
        elif self.aggregator == 'avg':
            # embedding: T(1, embedding_size)
            embedding = torch.sum(r_embedding, dim=1, keepdim=True) / r_embedding.shape[0]
            return self.predictor(embedding), self.predictor(r_embedding)
        elif self.aggregator == 'attention':
            # attention_score: T(num_sentences, num_relations)
            attention_score = torch.matmul(r_embedding, torch.t(weight))
            # attention_weight: T(num_sentences, num_relations)
            attention_weight = torch.nn.functional.softmax(attention_score, dim=0)
            # embedding: T(num_relations, embedding_size)
            embedding = torch.matmul(torch.transpose(attention_weight, 0, 1), r_embedding)
            # logit: T(num_relations, num_relations)
            logit = self.predictor(embedding)
            return torch.diag(logit).unsqueeze(0), self.predictor(r_embedding)
        else:
            assert False


class CodredCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_argument(self, parser):
        parser.add_argument('--seq_len', type=int, default=512)
        parser.add_argument('--aggregator', type=str, default='attention')
        parser.add_argument('--positive_only', action='store_true')
        parser.add_argument('--positive_ep_only', action='store_true')
        parser.add_argument('--no_doc_pair_supervision', action='store_true')
        parser.add_argument('--no_additional_marker', action='store_true')
        parser.add_argument('--mask_entity', action='store_true')
        parser.add_argument('--single_path', action='store_true')
        parser.add_argument('--dsre_only', action='store_true')
        parser.add_argument('--raw_only', action='store_true')
        parser.add_argument('--load_model_path', type=str, default=None)
        parser.add_argument('--train_file', type=str, default='../data/rawdata/train_dataset.json')
        parser.add_argument('--dev_file', type=str, default='../data/rawdata/dev_dataset.json')
        parser.add_argument('--dsre_file', type=str, default='../data/dsre_train_examples.json')

    def load_model(self):
        relations = json.load(open('../../../data/rawdata/relations.json'))
        relations.sort()
        self.relations = ['n/a'] + relations
        self.relation2id = dict()
        for index, relation in enumerate(self.relations):
            self.relation2id[relation] = index
        with self.trainer.cache():
            model = Codred(self.args, len(self.relations))
            if self.args.load_model_path:
                load_model(model, self.args.load_model_path)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.tokenizer = tokenizer

        self.sent_model = BertModel.from_pretrained('bert-base-cased')
        self.sent_model.eval()
        self.sent_model.cuda()

        return model

    def load_data(self):
        train_dataset = json.load(open(self.args.train_file))
        dev_dataset = json.load(open(self.args.dev_file))
        if self.args.positive_only:
            train_dataset = [d for d in train_dataset if d[3] != 'n/a']
            dev_dataset = [d for d in dev_dataset if d[3] != 'n/a']
        train_bags = place_train_data(train_dataset)
        dev_bags = place_dev_data(dev_dataset, self.args.single_path)
        if self.args.positive_ep_only:
            train_bags = [b for b in train_bags if b[1] != 'n/a']
            dev_bags = [b for b in dev_bags if 'n/a' not in b[1]]
        self.dsre_train_dataset = json.load(open(self.args.dsre_file))
        self.dsre_train_dataset = [d for i, d in enumerate(self.dsre_train_dataset) if i % 10 == 0]
        d = list()
        for i in range(len(self.dsre_train_dataset) // 8):
            d.append(self.dsre_train_dataset[8 * i:8 * i + 8])
        if self.args.raw_only:
            pass
        elif self.args.dsre_only:
            train_bags = d
        else:
            d.extend(train_bags)
            train_bags = d
        self.redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
        with self.trainer.once():
            self.train_logger = Logger(['train_loss', 'train_acc', 'train_pos_acc', 'train_dsre_acc'], self.trainer.writer, self.args.logging_steps, self.args.local_rank)
            self.dev_logger = Logger(['dev_mean_prec', 'dev_f1', 'dev_auc'], self.trainer.writer, 1, self.args.local_rank)
        return train_bags, dev_bags, dev_bags

    def collate_fn(self):
        return partial(collate_fn, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model)

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        with self.trainer.once():
            self.train_logger.log(train_loss=loss)
            _, f_prediction, f_logit = outputs
            for i in range(len(f_prediction)):
                if inputs['lst_rs'][i] is not None:
                    prediction = f_prediction[i]
                    logit = f_logit[i]
                    rs = extra['lst_rs'][i]
                    prediction, logit = tensor_to_obj(prediction, logit)
                    for p, score, gold in zip(prediction, logit, rs):
                        self.train_logger.log(train_acc=1 if p == gold else 0)
                        if gold > 0:
                            self.train_logger.log(train_pos_acc=1 if p == gold else 0)
                else:
                    prediction = f_prediction[i]
                    logit = f_logit[i]
                    dplabel = inputs['lst_dplabel'][i]
                    prediction, logit, dplabel = tensor_to_obj(prediction, logit, dplabel)
                    for p, l in zip(prediction, dplabel):
                        self.train_logger.log(train_dsre_acc=1 if p == l else 0)

    def on_train_epoch_end(self, epoch):
        pass

    def on_dev_epoch_start(self, epoch):
        self._prediction = list()

    def on_dev_step(self, step, inputs, extra, outputs):
        _, f_prediction, f_logit = outputs
        h, t, rs = extra['h'], extra['t'], extra['rs']
        for i in range(len(f_prediction)):
            prediction = f_prediction[i]
            logit = f_logit[i]
            prediction, logit = tensor_to_obj(prediction, logit)
            self._prediction.append([prediction[0], logit[0], h, t, rs])

    def on_dev_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        results = list()
        pred_result = list()
        facts = dict()
        for p, score, h, t, rs in self._prediction:
            rs = [self.relations[r] for r in rs]
            for i in range(1, len(score)):
                pred_result.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score[i]})
            results.append([h, rs, t, self.relations[p]])
            for r in rs:
                if r != 'n/a':
                    facts[(h, t, r)] = 1
        stat = eval_performance(facts, pred_result)
        with self.trainer.once():
            self.dev_logger.log(dev_mean_prec=stat['mean_prec'], dev_f1=stat['f1'], dev_auc=stat['auc'])
            json.dump(stat, open(f'output/dev-stat-{epoch}.json', 'w'))
            json.dump(results, open(f'output/dev-results-{epoch}.json', 'w'))
        return stat['f1']

    def process_train_data(self, data):
        # inputs = {
        #     'lst_input_ids': data[0],
        #     'lst_token_type_ids': data[1],
        #     'lst_attention_mask': data[2],
        #     'lst_dplabel': data[3],
        #     'lst_rs': data[4]}
        #
        # inputs_codre = {
        #     'lst_tokens_codre': data[6],
        #     'lst_intervals_codre': data[7],
        #     'lst_target_emb_codre': data[8],
        #     'lst_lst_emb_codre': data[9],
        #     'lst_dp_label_codre': data[10],
        #     'lst_rs_codre': data[11]
        # }
        return [], [], []

    def process_dev_data(self, data):
        inputs = {
            'lst_input_ids': data[0],
            'lst_token_type_ids': data[1],
            'lst_attention_mask': data[2]
        }
        return inputs, {'h': data[3], 'rs': data[4], 't': data[5]}


class CodredTrainer(Trainer):
    def __init__(self, callback: TrainerCallback):
        super().__init__(callback)

    def run(self):
        self.parse_args()
        self.set_env()
        with self.once():
            self.writer = torch.utils.tensorboard.SummaryWriter()
        self.set_model()
        self.load_data()
        if self.load_checkpoint is not None:
            self.restore_checkpoint(self.load_checkpoint, self.ignore_progress)
        best_performance = 0
        best_step = -1
        for epoch in range(self.epochs):
            if epoch < self.epochs_trained:
                continue
            with self.once():
                logging.info('epoch %d', epoch)
            if self.train:
                tr_loss, logging_loss = 0.0, 0.0
                self.model.zero_grad()
                self.model.train()
                self.callback.on_train_epoch_start(epoch)
                if self.local_rank >= 0:
                    self.train_sampler.set_epoch(epoch)
                for step, batch in enumerate(tqdm(self.train_dataloader, disable=self.local_rank > 0)):
                    continue

            # todo dev, test



def main():
    trainer = CodredTrainer(CodredCallback())
    trainer.run()


if __name__ == '__main__':
    main()
