import argparse
import os
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
import torch.nn as nn
import apex


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

def place_test_data(dataset, single_path):
    ep2d = dict()
    for data in dataset:
        key = data['h_id'] + '#' + data['t_id']
        doc1 = data['doc'][0]
        doc2 = data['doc'][1]
        label = 'n/a'
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
        pass
    try:
        target_tokens.remove(bound_tokens[1])
    except:
        pass
    target_tokens = ['[CLS]'] + target_tokens
    target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    if len(target_token_ids) < BERT_MAX_LEN:
        target_token_ids = target_token_ids + [0] * (BERT_MAX_LEN - len(target_token_ids))
    target_attention_mask = [1] * len(target_tokens) + [0] * (BERT_MAX_LEN - len(target_tokens))
    target_token_ids = torch.tensor(target_token_ids, dtype=torch.int64).unsqueeze(0).cuda()
    target_attention_mask = torch.tensor(target_attention_mask, dtype=torch.int64).unsqueeze(0).cuda()
    target_emb = sent_model(target_token_ids, target_attention_mask)[0][:, 0].data.cpu().numpy().tolist()

    lst_emb = []

    lst_token_ids = []
    lst_attention_mask = []

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

        lst_token_ids.append(tmp_token_ids)
        lst_attention_mask.append(tmp_attention_mask)

    if len(lst_token_ids) == 0:
        lst_emb = []
    else:
        token_ids = torch.cat(lst_token_ids, dim=0)
        attention_mask = torch.cat(lst_attention_mask, dim=0)

        batch_size = 32
        num_iter = len(token_ids) // batch_size + 1
        for itr in range(num_iter):
            if itr == num_iter - 1:
                lst_emb.append(sent_model(token_ids[batch_size*itr:], attention_mask[batch_size*itr:])[0][:, 0].data.cpu())
            else:
                lst_emb.append(sent_model(token_ids[batch_size*itr:batch_size*(itr+1)], attention_mask[batch_size*itr:batch_size*(itr+1)])[0][:, 0].data.cpu())

        lst_emb = torch.cat(lst_emb, dim=0).numpy().tolist()

    return ret, intervals, target_emb, lst_emb

def process_example(h, t, doc1, doc2, tokenizer, max_len, redisd, no_additional_marker, mask_entity, sent_model):
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
                k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2 = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
                tokens_codre.append((k1, k2))
                intervals_codre.append((intervals1, intervals2))
                target_emb_codre.append((target_emb1, target_emb2))
                lst_emb_codre.append((lst_emb1, lst_emb2))
                dplabel.append(relation2id[l])
            dplabel_t = torch.tensor(dplabel, dtype=torch.int64)
            rs_t = torch.tensor([r], dtype=torch.int64)

            lst_tokens_codre.append(tokens_codre)
            lst_intervals_codre.append(intervals_codre)
            lst_target_emb_codre.append(target_emb_codre)
            lst_lst_emb_codre.append(lst_emb_codre)
            lst_dp_label_codre.append(dplabel_t)
            lst_rs_codre.append(rs_t)
            lst_r_codre.append([r])
        else:
            examples = batch
            h_len = tokenizer.max_len_sentences_pair // 2 - 2
            t_len = tokenizer.max_len_sentences_pair - tokenizer.max_len_sentences_pair // 2 - 2
            _input_ids = list()
            _token_type_ids = list()
            _attention_mask = list()
            _rs = list()
            for idx, example in enumerate(examples):
                doc = json.loads(redisd.get(f'dsre-doc-{example[0]}'))
                _, h_start, h_end, t_start, t_end, r = example
                if r in relation2id:
                    r = relation2id[r]
                else:
                    r = 'n/a'
                h_1, h_2 = expand(h_start, h_end, len(doc), h_len)
                t_1, t_2 = expand(t_start, t_end, len(doc), t_len)
                h_tokens = doc[h_1:h_start] + ['[unused1]'] + doc[h_start:h_end] + ['[unused2]'] + doc[h_end:h_2]
                t_tokens = doc[t_1:t_start] + ['[unused3]'] + doc[t_start:t_end] + ['[unused4]'] + doc[t_end:t_2]
                h_token_ids = tokenizer.convert_tokens_to_ids(h_tokens)
                t_token_ids = tokenizer.convert_tokens_to_ids(t_tokens)
                input_ids = tokenizer.build_inputs_with_special_tokens(h_token_ids, t_token_ids)
                token_type_ids = tokenizer.create_token_type_ids_from_sequences(h_token_ids, t_token_ids)
                obj = tokenizer._pad({'input_ids': input_ids, 'token_type_ids': token_type_ids}, max_length=args.seq_len, padding_strategy='max_length')
                _input_ids.append(obj['input_ids'])
                _token_type_ids.append(obj['token_type_ids'])
                _attention_mask.append(obj['attention_mask'])
                _rs.append(r)
            input_ids_t = torch.tensor(_input_ids, dtype=torch.long)
            token_type_ids_t = torch.tensor(_token_type_ids, dtype=torch.long)
            attention_mask_t = torch.tensor(_attention_mask, dtype=torch.long)
            dplabel_t = torch.tensor(_rs, dtype=torch.long)
            rs_t = None
            r = None

            lst_input_ids_t.append(input_ids_t)
            lst_token_type_ids_t.append(token_type_ids_t)
            lst_attention_mask_t.append(attention_mask_t)
            lst_dplabel_t.append(dplabel_t)
            lst_rs_t.append(rs_t)
            lst_r.append([r])
    return lst_input_ids_t, lst_token_type_ids_t, lst_attention_mask_t, lst_dplabel_t, lst_rs_t, lst_r,\
           lst_tokens_codre, lst_intervals_codre, lst_target_emb_codre, lst_lst_emb_codre, lst_dp_label_codre, lst_rs_codre, lst_r_codre


def collate_fn_infer(batch, args, relation2id, tokenizer, redisd, sent_model):
    assert len(batch) == 1
    batch = batch[0]
    h, t = batch[0].split('#')
    rs = [relation2id[r] for r in batch[1]]
    dps = batch[2]

    tokens_codre = list()
    intervals_codre = list()
    target_emb_codre = list()
    lst_emb_codre = list()

    for doc1, doc2, l in dps:
        k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2 = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
        tokens_codre.append((k1, k2))
        intervals_codre.append((intervals1, intervals2))
        target_emb_codre.append((target_emb1, target_emb2))
        lst_emb_codre.append((lst_emb1, lst_emb2))

    return tokens_codre, intervals_codre, target_emb_codre, lst_emb_codre, h, rs, t


def collate_fn_test(batch, args, relation2id, tokenizer, redisd, sent_model):
    assert len(batch) == 1
    batch = batch[0]
    h, t = batch[0].split('#')
    rs = [relation2id[r] for r in batch[1]]
    dps = batch[2]

    tokens_codre = list()
    intervals_codre = list()
    target_emb_codre = list()
    lst_emb_codre = list()

    for doc1, doc2, l in dps:
        k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2 = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
        tokens_codre.append((k1, k2))
        intervals_codre.append((intervals1, intervals2))
        target_emb_codre.append((target_emb1, target_emb2))
        lst_emb_codre.append((lst_emb1, lst_emb2))

    return tokens_codre, intervals_codre, target_emb_codre, lst_emb_codre, h, rs, t

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
    
    def forward(self, lst_input_ids, lst_token_type_ids, lst_attention_mask, lst_dplabel=None, lst_rs=None, need_dplogit=False):
        f_loss = 0.
        f_prediction = []
        f_logit = []
        f_dp_logit = []
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
            f_dp_logit.append(dp_logit)
        f_loss /= len(lst_input_ids)
        if need_dplogit:
            return f_loss, f_prediction, f_logit, f_dp_logit
        else:
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
        parser.add_argument('--load_selector_path', type=str, default=None)
        parser.add_argument('--train_file', type=str, default='../data/rawdata/train_dataset.json')
        parser.add_argument('--dev_file', type=str, default='../data/rawdata/dev_dataset.json')
        parser.add_argument('--test_file', type=str, default='../data/rawdata/test_dataset.json')
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
            selector = LSTMSelector(768, 512, 512)
            if self.args.load_model_path:
                load_model(model, self.args.load_model_path)
            if self.args.load_selector_path:
                load_model(selector, self.args.load_selector_path)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.tokenizer = tokenizer

        self.sent_model = BertModel.from_pretrained('bert-base-cased')
        self.sent_model.eval()
        self.sent_model.cuda()

        return model, selector

    def load_data(self):
        train_dataset = json.load(open(self.args.train_file))
        dev_dataset = json.load(open(self.args.dev_file))
        test_dataset = json.load(open(self.args.test_file))
        if self.args.positive_only:
            train_dataset = [d for d in train_dataset if d[3] != 'n/a']
            dev_dataset = [d for d in dev_dataset if d[3] != 'n/a']
            test_dataset = [d for d in test_dataset if d[3] != 'n/a']
        train_bags = place_train_data(train_dataset)
        dev_bags = place_dev_data(dev_dataset, self.args.single_path)
        test_bags = place_test_data(test_dataset, self.args.single_path)
        if self.args.positive_ep_only:
            train_bags = [b for b in train_bags if b[1] != 'n/a']
            dev_bags = [b for b in dev_bags if 'n/a' not in b[1]]
            test_bags = [b for b in test_bags if 'n/a' not in b[1]]
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
        self.redisd = redis.Redis(host='localhost', port=6379, decode_responses=True, health_check_interval=30)
        with self.trainer.once():
            self.train_logger = Logger(['train_loss', 'train_acc', 'train_pos_acc', 'train_dsre_acc'], self.trainer.writer, self.args.logging_steps, self.args.local_rank)
            self.dev_logger = Logger(['dev_mean_prec', 'dev_f1', 'dev_auc'], self.trainer.writer, 1, self.args.local_rank)
            self.test_logger = Logger(['test_mean_prec', 'test_f1', 'test_auc'], self.trainer.writer, 1, self.args.local_rank)
        return train_bags, dev_bags, test_bags

    def collate_fn(self):
        return partial(collate_fn, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model), partial(collate_fn_test, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model)

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        with self.trainer.once():
            self.train_logger.log(train_loss=loss)
            _, f_prediction, f_logit, _ = outputs
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

    def on_test_epoch_start(self, epoch):
        self._prediction = list()
        pass

    def on_test_step(self, step, inputs, extra, outputs):
        _, f_prediction, f_logit = outputs
        h, t, rs = extra['h'], extra['t'], extra['rs']
        for i in range(len(f_prediction)):
            prediction = f_prediction[i]
            logit = f_logit[i]
            prediction, logit = tensor_to_obj(prediction, logit)
            self._prediction.append([prediction[0], logit[0], h, t, rs])

    def on_test_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        results = list()
        pred_result = list()
        facts = dict()
        out_results = list()
        coda_file = dict()
        coda_file['setting'] = 'closed'
        for p, score, h, t, rs in self._prediction:
            rs = [self.relations[r] for r in rs]
            for i in range(1, len(score)):
                pred_result.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score[i]})
                out_results.append({'h_id':str(h), "t_id":str(t), "relation": str(self.relations[i]), "score": float(score[i])})
            results.append([h, rs, t, self.relations[p]])
            for r in rs:
                if r != 'n/a':
                    facts[(h, t, r)] = 1
        coda_file['predictions'] = out_results
        with self.trainer.once():
            json.dump(results, open(f'output/test-results-{epoch}.json', 'w'))
            json.dump(coda_file, open(f'output/test-codalab-results-{epoch}.json', 'w'))

    def process_train_data(self, data):
        inputs = {
            'lst_input_ids': data[0],
            'lst_token_type_ids': data[1],
            'lst_attention_mask': data[2],
            'lst_dplabel': data[3],
            'lst_rs': data[4]}

        inputs_codre = {
            'lst_tokens_codre': data[6],
            'lst_intervals_codre': data[7],
            'lst_target_emb_codre': data[8],
            'lst_lst_emb_codre': data[9],
            'lst_dplabel_codre': data[10],
            'lst_rs_codre': data[11]
        }
        return inputs, inputs_codre, {'lst_rs': data[5], 'lst_rs_codre': data[12]}

    def process_dev_data(self, data):
        inputs = {
            'lst_tokens_codre': data[0],
            'lst_intervals_codre': data[1],
            'lst_target_emb_codre': data[2],
            'lst_lst_emb_codre': data[3]
        }
        return inputs, {'h': data[4], 'rs': data[5], 't': data[6]}

    def process_test_data(self, data):
        inputs = {
            'lst_tokens_codre': data[0],
            'lst_intervals_codre': data[1],
            'lst_target_emb_codre': data[2],
            'lst_lst_emb_codre': data[3]
        }
        return inputs, {'h': data[4], 'rs': data[5], 't': data[6]}


class CodredTrainer(Trainer):
    def __init__(self, callback: TrainerCallback):
        super().__init__(callback)

    def parse_args(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--dev', action='store_true')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
        self.parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
        self.parser.add_argument("--learning_rate", default=3e-5, type=float)
        self.parser.add_argument("--selector_learning_rate", default=3e-3, type=float)
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        self.parser.add_argument("--weight_decay", default=0.0, type=float)
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float)
        self.parser.add_argument("--epochs", default=2, type=int)
        self.parser.add_argument("--warmup_ratio", default=0.1, type=float)
        self.parser.add_argument("--logging_steps", type=int, default=500)
        self.parser.add_argument("--save_steps", type=int, default=10000)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--num_workers", type=int, default=0)
        self.parser.add_argument("--local_rank", type=int, default=-1)
        self.parser.add_argument("--fp16", action="store_true")
        self.parser.add_argument("--fp16_opt_level", type=str, default="O1")
        self.parser.add_argument("--no_cuda", action="store_true")
        self.parser.add_argument("--load_checkpoint", default=None, type=str)
        self.parser.add_argument("--ignore_progress", action='store_true')
        self.parser.add_argument("--dataset_ratio", type=float, default=1.0)
        self.parser.add_argument("--no_save", action="store_true")
        self.parser.add_argument("--num_sentences", default=3, type=int)
        self.parser.add_argument("--lam_notnone", default=10, type=int)

        self.callback.on_argument(self.parser)
        self.args = self.parser.parse_args()
        keys = list(self.args.__dict__.keys())
        for key in keys:
            value = getattr(self.args, key)
            if type(value) == str and os.path.exists(value):
                setattr(self.args, key, os.path.abspath(value))
        if not self.args.train:
            self.args.epochs = 1
        self.train = self.args.train
        self.dev = self.args.dev
        self.test = self.args.test
        self.debug = self.args.debug
        self.per_gpu_train_batch_size = self.args.per_gpu_train_batch_size
        self.per_gpu_eval_batch_size = self.args.per_gpu_eval_batch_size
        self.learning_rate = self.args.learning_rate
        self.selector_learning_rate = self.args.selector_learning_rate
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        self.weight_decay = self.args.weight_decay
        self.adam_epsilon = self.args.adam_epsilon
        self.max_grad_norm = self.args.max_grad_norm
        self.epochs = self.args.epochs
        self.warmup_ratio = self.args.warmup_ratio
        self.logging_steps = self.args.logging_steps
        self.save_steps = self.args.save_steps
        self.seed = self.args.seed
        self.num_workers = self.args.num_workers
        self.local_rank = self.args.local_rank
        self.fp16 = self.args.fp16
        self.fp16_opt_level = self.args.fp16_opt_level
        self.no_cuda = self.args.no_cuda
        self.load_checkpoint = self.args.load_checkpoint
        self.ignore_progress = self.args.ignore_progress
        self.dataset_ratio = self.args.dataset_ratio
        self.no_save = self.args.no_save
        self.callback.args = self.args

    def set_model(self):
        self.model, self.selector = self.callback.load_model()
        self.model.to(self.device)
        self.selector.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": self.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.selector_optimizer = torch.optim.AdamW(self.selector.parameters(), lr=self.selector_learning_rate, eps=self.adam_epsilon)

        if self.fp16:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)
            self.selector, self.selector_optimizer = apex.amp.initialize(self.selector, self.selector_optimizer, opt_level=self.fp16_opt_level)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.selector = torch.nn.DataParallel(self.selector)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            self.selector = torch.nn.parallel.DistributedDataParallel(self.selector, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def save_checkpoint(self):
        if self.no_save:
            return
        output_dir = os.path.join('output', "checkpoint-{}".format(self.train_step))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        selector_to_save = self.selector.module if hasattr(self.selector, "module") else self.selector
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        torch.save(selector_to_save.state_dict(), os.path.join(output_dir, 'selector_pytorch_model.bin'))
        torch.save(self.selector_optimizer.state_dict(), os.path.join(output_dir, "selector_optimizer.pt"))

        self.callback.on_save(output_dir)

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
        BERT_MAX_LEN = 512
        for epoch in range(self.epochs):
            if epoch < self.epochs_trained:
                continue
            with self.once():
                logging.info('epoch %d', epoch)
            if self.train:
                tr_loss, tr_s_loss, logging_loss, logging_s_loss = 0.0, 0.0, 0.0, 0.0
                self.model.zero_grad()
                self.model.train()
                self.selector.zero_grad()
                self.selector.train()
                self.callback.on_train_epoch_start(epoch)
                if self.local_rank >= 0:
                    self.train_sampler.set_epoch(epoch)
                for step, batch in enumerate(tqdm(self.train_dataloader, disable=self.local_rank > 0)):
                    if step < self.steps_trained_in_current_epoch:
                        continue
                    inputs, inputs_codre, extra = self.callback.process_train_data(batch)

                    num_dsre = len(inputs['lst_input_ids'])
                    num_codre = len(inputs_codre['lst_tokens_codre'])

                    if num_codre != 0:
                        lst_log_probs = []
                        for i in range(len(inputs_codre['lst_target_emb_codre'])):

                            input_ids = list()
                            token_type_ids = list()
                            attention_mask = list()

                            tokens = inputs_codre['lst_tokens_codre'][i]
                            intervals = inputs_codre['lst_intervals_codre'][i]
                            target_emb = inputs_codre['lst_target_emb_codre'][i]
                            lst_emb = inputs_codre['lst_lst_emb_codre'][i]

                            target_emb = torch.Tensor(target_emb) # (num path x 2) x embedding size
                            target_emb = target_emb.reshape(-1, 768).to(self.device)

                            lst_emb, ctx_len = pad_to_max_ns(lst_emb) # (num path x 2) x max num sent. x embedding size
                            lst_emb = torch.stack(lst_emb).to(self.device)

                            num_selection = self.args.num_sentences
                            selection, dist, log_probs = self.selector(target_emb, lst_emb, None, ctx_len, num_selection)
                            log_probs = log_probs.reshape(-1, 2).sum(dim=1)
                            lst_log_probs.append(log_probs)

                            for idx, (token, interval) in enumerate(zip(tokens, intervals)):
                                target_st = 0
                                target_en = -1
                                for l in range(len(interval[0])):
                                    if target_st != interval[0][l][0]:
                                        target_en = interval[0][l][0]
                                        try:
                                            for j in range(10):
                                                if token[0][target_en + j] == '[unused2]':
                                                    target_en += j+1
                                                    interval[0][l][0] += j+1
                                                    break
                                        except:
                                            pass
                                        break
                                    else:
                                        target_st = interval[0][l][1]
                                if target_en == -1:
                                    target_en = len(token[0])

                                k1 = token[0][target_st:target_en]

                                for k in range(min(num_selection, len(interval[0]))):
                                    tmp_sel = selection[k][2*idx]
                                    st = interval[0][tmp_sel][0]
                                    en = interval[0][tmp_sel][1]
                                    k1 = k1 + token[0][st:en]

                                if len(k1) > BERT_MAX_LEN // 2 - 2:
                                    k1 = k1[:BERT_MAX_LEN // 2 - 2]

                                target_st = 0
                                target_en = -1
                                for l in range(len(interval[1])):
                                    if target_st != interval[1][l][0]:
                                        target_en = interval[1][l][0]
                                        try:
                                            for j in range(10):
                                                if token[1][target_en + j] == '[unused4]':
                                                    target_en += j+1
                                                    interval[1][l][0] += j+1
                                                    break
                                        except:
                                            pass
                                        break
                                    else:
                                        target_st = interval[1][l][1]
                                if target_en == -1:
                                    target_en = len(token[1])

                                k2 = token[1][target_st:target_en]

                                for k in range(min(num_selection, len(interval[1]))):
                                    tmp_sel = selection[k][2 * idx + 1]
                                    st = interval[1][tmp_sel][0]
                                    en = interval[1][tmp_sel][1]
                                    k2 = k2 + token[1][st:en]

                                if len(k2) > BERT_MAX_LEN // 2 - 1:
                                    k2 = k2[:BERT_MAX_LEN // 2 - 1]

                                tmp_token = ['[CLS]'] + k1 + ['[SEP]'] + k2 + ['[SEP]']

                                tmp_token_ids = self.callback.tokenizer.convert_tokens_to_ids(tmp_token)
                                if len(tmp_token_ids) < BERT_MAX_LEN:
                                    tmp_token_ids = tmp_token_ids + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
                                tmp_attention_mask = [1] * len(tmp_token) + [0] * (BERT_MAX_LEN - len(tmp_token))
                                tmp_token_type_ids = [0] * (len(k1) + 2) + [1] * (len(k2) + 1) + [0] * (BERT_MAX_LEN - len(tmp_token))

                                input_ids.append(tmp_token_ids)
                                token_type_ids.append(tmp_token_type_ids)
                                attention_mask.append(tmp_attention_mask)

                            input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
                            token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
                            attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)

                            inputs['lst_input_ids'].append(input_ids_t)
                            inputs['lst_token_type_ids'].append(token_type_ids_t)
                            inputs['lst_attention_mask'].append(attention_mask_t)

                        inputs['lst_dplabel'] = inputs['lst_dplabel'] + inputs_codre['lst_dplabel_codre']
                        inputs['lst_rs'] = inputs['lst_rs'] + inputs_codre['lst_rs_codre']

                        extra['lst_rs'] = extra['lst_rs'] + extra['lst_rs_codre']

                    outputs = self.model(**inputs, need_dplogit=True)
                    loss = outputs[0]

                    if num_codre != 0:
                        s_loss = 0.0
                        for idx, log_probs in enumerate(lst_log_probs):
                            pred_prob = torch.softmax(outputs[3][num_dsre + idx].detach(), dim=1)
                            dplabel = inputs['lst_dplabel'][num_dsre + idx]
                            for idx_path, log_prob in enumerate(log_probs):
                                pred_prob_tmp = pred_prob[idx_path].clone()
                                pred_prob_tmp[dplabel[idx_path]] = float('-inf')
                                pred_prob_max = pred_prob_tmp.max()
                                reward = (pred_prob[idx_path][dplabel[idx_path]] - pred_prob_max) / (pred_prob[idx_path][dplabel[idx_path]] + 1e-7)
                                reward = max(0, reward)
                                if dplabel[idx_path] != 0:
                                    reward = reward * self.args.lam_notnone
                                s_loss = s_loss - reward * log_prob

                    if self.n_gpu > 1:
                        loss = loss.mean()
                        if num_codre != 0:
                            s_loss = s_loss.mean()
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                        if num_codre != 0:
                            s_loss = s_loss / self.gradient_accumulation_steps
                    if self.local_rank < 0 or (step + 1) % self.gradient_accumulation_steps == 0:
                        if self.fp16:
                            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                            if num_codre != 0:
                                with apex.amp.scale_loss(s_loss, self.selector_optimizer) as scaled_s_loss:
                                    scaled_s_loss.backward()
                        else:
                            loss.backward()
                            if num_codre != 0:
                                s_loss.backward()
                    else:
                        with self.model.no_sync():
                            if self.fp16:
                                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                    scaled_loss.backward()
                                if num_codre != 0:
                                    with apex.amp.scale_loss(s_loss, self.selector_optimizer) as scaled_s_loss:
                                        scaled_s_loss.backward()
                            else:
                                loss.backward()
                                if num_codre != 0:
                                    s_loss.backward()
                    tr_loss += loss.item()
                    if num_codre != 0:
                        tr_s_loss += s_loss.item()
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        if self.fp16:
                            torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), self.max_grad_norm)
                            if num_codre != 0:
                                torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.selector_optimizer), self.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            if num_codre != 0:
                                torch.nn.utils.clip_grad_norm_(self.selector.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        if num_codre != 0:
                            self.selector_optimizer.step()
                        self.model.zero_grad()
                        self.selector.zero_grad()
                        self.train_step += 1
                        with self.once():
                            if self.train_step % self.logging_steps == 0:
                                self.writer.add_scalar("lr", self.scheduler.get_lr()[0], self.train_step)
                                self.writer.add_scalar("loss", (tr_loss - logging_loss) / self.logging_steps,
                                                       self.train_step)
                                if num_codre != 0:
                                    self.writer.add_scalar("s_loss", (tr_s_loss - logging_s_loss) / self.logging_steps,
                                                           self.train_step)
                                logging_loss = tr_loss
                                if num_codre != 0:
                                    logging_s_loss = tr_s_loss
                            if self.train_step % self.save_steps == 0:
                                self.save_checkpoint()
                    self.callback.on_train_step(step, self.train_step, inputs, extra, loss.item(), outputs)
                with self.once():
                    self.save_checkpoint()
                self.callback.on_train_epoch_end(epoch)
            if self.dev:
                with torch.no_grad():
                    self.model.eval()
                    self.callback.on_dev_epoch_start(epoch)
                    for step, batch in enumerate(tqdm(self.dev_dataloader, disable=self.local_rank > 0)):
                        inputs_codre, extra = self.callback.process_dev_data(batch)

                        input_ids = list()
                        token_type_ids = list()
                        attention_mask = list()

                        tokens = inputs_codre['lst_tokens_codre']
                        intervals = inputs_codre['lst_intervals_codre']
                        target_emb = inputs_codre['lst_target_emb_codre']
                        lst_emb = inputs_codre['lst_lst_emb_codre']

                        if len(tokens) == 0:
                            h, t, rs = extra['h'], extra['t'], extra['rs']
                            self.callback._prediction.append([0, [10] + [-1] * 276, h, t, rs])
                            print('no token!')
                            continue

                        target_emb = torch.Tensor(target_emb) # (num path x 2) x embedding size
                        target_emb = target_emb.reshape(-1, 768).to(self.device)

                        lst_emb, ctx_len = pad_to_max_ns(lst_emb) # (num path x 2) x max num sent. x embedding size
                        lst_emb = torch.stack(lst_emb).to(self.device)

                        num_selection = self.args.num_sentences
                        selection, dist, log_probs = self.selector(target_emb, lst_emb, None, ctx_len, num_selection)
                        log_probs = log_probs.reshape(-1, 2).sum(dim=1)

                        for idx, (token, interval) in enumerate(zip(tokens, intervals)):
                            target_st = 0
                            target_en = -1
                            for l in range(len(interval[0])):
                                if target_st != interval[0][l][0]:
                                    target_en = interval[0][l][0]
                                    try:
                                        for j in range(10):
                                            if token[0][target_en + j] == '[unused2]':
                                                target_en += j + 1
                                                interval[0][l][0] += j + 1
                                                break
                                    except:
                                        pass
                                    break
                                else:
                                    target_st = interval[0][l][1]
                            if target_en == -1:
                                target_en = len(token[0])

                            k1 = token[0][target_st:target_en]

                            for k in range(min(num_selection, len(interval[0]))):
                                tmp_sel = selection[k][2*idx]
                                st = interval[0][tmp_sel][0]
                                en = interval[0][tmp_sel][1]
                                k1 = k1 + token[0][st:en]

                            if len(k1) > BERT_MAX_LEN // 2 - 2:
                                k1 = k1[:BERT_MAX_LEN // 2 - 2]

                            target_st = 0
                            target_en = -1
                            for l in range(len(interval[1])):
                                if target_st != interval[1][l][0]:
                                    target_en = interval[1][l][0]
                                    try:
                                        for j in range(10):
                                            if token[1][target_en + j] == '[unused4]':
                                                target_en += j + 1
                                                interval[1][l][0] += j + 1
                                                break
                                    except:
                                        pass
                                    break
                                else:
                                    target_st = interval[1][l][1]
                            if target_en == -1:
                                target_en = len(token[1])

                            k2 = token[1][target_st:target_en]

                            for k in range(min(num_selection, len(interval[1]))):
                                tmp_sel = selection[k][2 * idx + 1]
                                st = interval[1][tmp_sel][0]
                                en = interval[1][tmp_sel][1]
                                k2 = k2 + token[1][st:en]

                            if len(k2) > BERT_MAX_LEN // 2 - 1:
                                k2 = k2[:BERT_MAX_LEN // 2 - 1]

                            tmp_token = ['[CLS]'] + k1 + ['[SEP]'] + k2 + ['[SEP]']

                            tmp_token_ids = self.callback.tokenizer.convert_tokens_to_ids(tmp_token)
                            if len(tmp_token_ids) < BERT_MAX_LEN:
                                tmp_token_ids = tmp_token_ids + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
                            tmp_attention_mask = [1] * len(tmp_token) + [0] * (BERT_MAX_LEN - len(tmp_token))
                            tmp_token_type_ids = [0] * (len(k1) + 2) + [1] * (len(k2) + 1) + [0] * (BERT_MAX_LEN - len(tmp_token))

                            input_ids.append(tmp_token_ids)
                            token_type_ids.append(tmp_token_type_ids)
                            attention_mask.append(tmp_attention_mask)

                        input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
                        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
                        attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)

                        input_ids_t = torch.stack([input_ids_t])
                        token_type_ids_t = torch.stack([token_type_ids_t])
                        attention_mask_t = torch.stack([attention_mask_t])

                        inputs = {
                            'lst_input_ids': input_ids_t,
                            'lst_token_type_ids': token_type_ids_t,
                            'lst_attention_mask': attention_mask_t
                        }

                        outputs = self.model(**inputs)
                        self.callback.on_dev_step(step, inputs, extra, outputs)
                    performance = self.callback.on_dev_epoch_end(epoch)
                    if performance > best_performance:
                        best_performance = performance
                        best_step = self.train_step
        if self.test:
            with torch.no_grad():
                if best_step > 0 and self.train:
                    self.restore_checkpoint(os.path.join('output', "checkpoint-{}".format(best_step)))
                self.model.eval()
                self.callback.on_test_epoch_start(epoch)
                for step, batch in enumerate(tqdm(self.test_dataloader, disable=self.local_rank > 0)):
                    inputs_codre, extra = self.callback.process_dev_data(batch)

                    input_ids = list()
                    token_type_ids = list()
                    attention_mask = list()

                    tokens = inputs_codre['lst_tokens_codre']
                    intervals = inputs_codre['lst_intervals_codre']
                    target_emb = inputs_codre['lst_target_emb_codre']
                    lst_emb = inputs_codre['lst_lst_emb_codre']

                    if len(tokens) == 0:
                        h, t, rs = extra['h'], extra['t'], extra['rs']
                        self.callback._prediction.append([0, [10] + [-1] * 276, h, t, rs])
                        print('no token!')
                        continue

                    target_emb = torch.Tensor(target_emb)  # (num path x 2) x embedding size
                    target_emb = target_emb.reshape(-1, 768).to(self.device)

                    lst_emb, ctx_len = pad_to_max_ns(lst_emb)  # (num path x 2) x max num sent. x embedding size
                    lst_emb = torch.stack(lst_emb).to(self.device)

                    num_selection = self.args.num_sentences
                    selection, dist, log_probs = self.selector(target_emb, lst_emb, None, ctx_len, num_selection)
                    log_probs = log_probs.reshape(-1, 2).sum(dim=1)

                    for idx, (token, interval) in enumerate(zip(tokens, intervals)):
                        target_st = 0
                        target_en = -1
                        for l in range(len(interval[0])):
                            if target_st != interval[0][l][0]:
                                target_en = interval[0][l][0]
                                try:
                                    for j in range(10):
                                        if token[0][target_en + j] == '[unused2]':
                                            target_en += j + 1
                                            interval[0][l][0] += j + 1
                                            break
                                except:
                                    pass
                                break
                            else:
                                target_st = interval[0][l][1]
                        if target_en == -1:
                            target_en = len(token[0])

                        k1 = token[0][target_st:target_en]

                        for k in range(min(num_selection, len(interval[0]))):
                            tmp_sel = selection[k][2 * idx]
                            st = interval[0][tmp_sel][0]
                            en = interval[0][tmp_sel][1]
                            k1 = k1 + token[0][st:en]

                        if len(k1) > BERT_MAX_LEN // 2 - 2:
                            k1 = k1[:BERT_MAX_LEN // 2 - 2]

                        target_st = 0
                        target_en = -1
                        for l in range(len(interval[1])):
                            if target_st != interval[1][l][0]:
                                target_en = interval[1][l][0]
                                try:
                                    for j in range(10):
                                        if token[1][target_en + j] == '[unused4]':
                                            target_en += j + 1
                                            interval[1][l][0] += j + 1
                                            break
                                except:
                                    pass
                                break
                            else:
                                target_st = interval[1][l][1]
                        if target_en == -1:
                            target_en = len(token[1])

                        k2 = token[1][target_st:target_en]

                        for k in range(min(num_selection, len(interval[1]))):
                            tmp_sel = selection[k][2 * idx + 1]
                            st = interval[1][tmp_sel][0]
                            en = interval[1][tmp_sel][1]
                            k2 = k2 + token[1][st:en]

                        if len(k2) > BERT_MAX_LEN // 2 - 1:
                            k2 = k2[:BERT_MAX_LEN // 2 - 1]

                        tmp_token = ['[CLS]'] + k1 + ['[SEP]'] + k2 + ['[SEP]']

                        tmp_token_ids = self.callback.tokenizer.convert_tokens_to_ids(tmp_token)
                        if len(tmp_token_ids) < BERT_MAX_LEN:
                            tmp_token_ids = tmp_token_ids + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
                        tmp_attention_mask = [1] * len(tmp_token) + [0] * (BERT_MAX_LEN - len(tmp_token))
                        tmp_token_type_ids = [0] * (len(k1) + 2) + [1] * (len(k2) + 1) + [0] * (
                                    BERT_MAX_LEN - len(tmp_token))

                        input_ids.append(tmp_token_ids)
                        token_type_ids.append(tmp_token_type_ids)
                        attention_mask.append(tmp_attention_mask)

                    input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
                    token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
                    attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)

                    input_ids_t = torch.stack([input_ids_t])
                    token_type_ids_t = torch.stack([token_type_ids_t])
                    attention_mask_t = torch.stack([attention_mask_t])

                    inputs = {
                        'lst_input_ids': input_ids_t,
                        'lst_token_type_ids': token_type_ids_t,
                        'lst_attention_mask': attention_mask_t
                    }

                    outputs = self.model(**inputs)
                    self.callback.on_test_step(step, inputs, extra, outputs)
                self.callback.on_test_epoch_end(epoch)
        with self.once():
            self.writer.close()
        json.dump(True, open('output/f.json', 'w'))


class LSTMSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_dim, fn_activate='tanh'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(self.in_dim, self.hidden_dim)

        self.mlp_dim = mlp_dim

        self.drop_out = nn.Dropout(0.5)

        self.mlp1 = nn.Linear(self.hidden_dim + self.in_dim, self.mlp_dim)

        if fn_activate == 'relu':
            self.fn_activate1 = nn.LeakyReLU(0.2, True)
        elif fn_activate == 'tanh':
            self.fn_activate1 = nn.Tanh()
        elif fn_activate == 'relu6':
            self.fn_activate1 = nn.ReLU6()
        elif fn_activate == 'silu':
            self.fn_activate1 = nn.SiLU()
        elif fn_activate == 'hardtanh':
            self.fn_activate1 = nn.Hardtanh()

        self.mlp2 = nn.Linear(self.mlp_dim, 1)

        self.fn_activate2 = nn.Sigmoid()

    def forward(self, target_emb: torch.Tensor, ctx_emb: torch.Tensor, target_len, ctx_len, n_step):
        outputs = []
        dist = []

        lstm_in = target_emb  # bs x dim

        bs = lstm_in.size(0)
        ns = ctx_emb.size(1)

        h_0 = torch.zeros((bs, self.hidden_dim))
        c_0 = torch.zeros((bs, self.hidden_dim))

        # mask for selected sentences
        mask = torch.zeros((bs, ns))
        for i in range(bs):
            if ctx_len[i] < ns:
                mask[i, ctx_len[i]:] = -100000

        log_probs = torch.zeros((bs))
        h_0 = h_0.to(target_emb.device)
        c_0 = c_0.to(target_emb.device)
        mask = mask.to(target_emb.device)
        log_probs = log_probs.to(target_emb.device)
        lstm_state = (h_0, c_0)

        for _ in range(n_step):
            lstm_in = self.drop_out(lstm_in)
            h, c = self.lstm_cell(lstm_in, lstm_state)  # h: bs x hidden_dim

            _ctx_emb = self.drop_out(ctx_emb)
            h = self.drop_out(h)

            sc = self.mlp1(torch.cat([_ctx_emb, h.unsqueeze(1).expand((-1, ns, -1))], dim=-1))  # bs x ns x mlp_dim
            sc = self.drop_out(sc)
            sc = self.mlp2(self.fn_activate1(sc))  # bs x ns x 1
            # sc = self.fn_activate2(sc.squeeze())
            sc = sc.squeeze()
            sc = sc + mask

            if self.training:
                probs = torch.softmax(sc, dim=-1)  # bs x ns
                probs = torch.distributions.Categorical(probs=probs)
                # out = sc.max(dim=-1)[1] # bs x 1: index of selected sentence in this step
                out = probs.sample()
                log_probs = log_probs + probs.log_prob(out)
                dist.append(probs)
                outputs.append(out)
            else:
                out = sc.max(dim=-1)[1]  # bs x 1: index of selected sentence in this step
                outputs.append(out)

            for i in range(bs):
                mask[i, out[i]] = -100000  # mask selected sentent

            lstm_in = torch.gather(ctx_emb, dim=1, index=out.unsqueeze(1).unsqueeze(2).expand(bs, 1, self.in_dim))
            lstm_in = lstm_in.squeeze(1)
            lstm_state = (h, c)

        return outputs, dist, log_probs


def pad_to_max_ns(ctx_augm_emb):
    max_ns = 0
    ctx_augm_emb_paded = []
    ctx_len = []
    for path in ctx_augm_emb:
        for doc in path:
            max_ns = max(max_ns, len(doc))

    for path in ctx_augm_emb:
        for doc in path:
            pad = torch.zeros((max_ns, 768))
            if len(doc) == 0:
                ctx_augm_emb_paded.append(pad)
                ctx_len.append(0)
            else:
                doc = torch.Tensor(doc).squeeze(1)
                ctx_len.append(len(doc))
                if len(doc) < max_ns:
                    pad[:len(doc), :] = doc
                    ctx_augm_emb_paded.append(pad)
                else:
                    ctx_augm_emb_paded.append(doc)
    return ctx_augm_emb_paded, ctx_len



def main():
    trainer = CodredTrainer(CodredCallback())
    trainer.run()


if __name__ == '__main__':
    main()
