### from concurrent.futures.thread import _threads_queues
import json
import random
from functools import partial
import pdb
import numpy as np
import redis
import sklearn
import torch
from eveliver import (Logger, load_model, tensor_to_obj)
from trainer_save import Trainer, TrainerCallback, LSTMSelector
from transformers import AutoTokenizer, BertModel
# from matrix_transformer import Encoder as MatTransformer
from torch import nn
import torch.nn.functional as F
# from buffer import Buffer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME, contrastive_pair, check_htb_debug, complete_h_t_debug
from utils import complete_h_t, check_htb, check_htb_debug
from utils import CLS_TOKEN_ID, SEP_TOKEN_ID, H_START_MARKER_ID, H_END_MARKER_ID, T_END_MARKER_ID, T_START_MARKER_ID
# from sbert_wk import sbert
import os

def eval_performance(facts, pred_result):
    sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec = []
    rec = []
    correct = 0
    total = len(facts)
    #pdb.set_trace()
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


# def process(tokenizer, h, t, doc0, doc1):
#
#     ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
#     b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
#     max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
#     cnt, batches = 0, []
#     d = []
#
#     def fix_entity(doc, ht_markers, b_markers):
#         markers = ht_markers + b_markers
#         markers_pos = []
#         if list(set(doc).intersection(set(markers))):
#             for marker in markers:
#                 try:
#                     pos = doc.index(marker)
#                     markers_pos.append((pos, marker))
#                 except ValueError as e:
#                     continue
#
#         idx = 0
#         while idx <= len(markers_pos)-1:
#             try:
#                 assert (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) % 2 == 1) and (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) - int(markers_pos[idx+1][1].replace("[unused", "").replace("]", "")) == -1)
#                 entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
#                 while "." in entity_name:
#                     assert doc[markers_pos[idx][0] + entity_name.index(".") + 1] == "."
#                     doc[markers_pos[idx][0] + entity_name.index(".") + 1] = "|"
#                     entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
#                 idx += 2
#             except:
#                 #pdb.set_trace()
#                 idx += 1
#                 continue
#         return doc
#
#     d0 = fix_entity(doc0, ht_markers, b_markers)
#     d1 = fix_entity(doc1, ht_markers, b_markers)
#
#     for di in [d0, d1]:
#         d.extend(di)
#     d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, cnt=cnt, hard=False, docid=0)
#     d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, cnt=cnt, hard=False, docid=1)
#     dbuf = Buffer()
#     dbuf.blocks = d0_buf.blocks + d1_buf.blocks
#     for blk in dbuf:
#         if list(set(tokenizer.convert_tokens_to_ids(ht_markers)).intersection(set(blk.ids))):
#             blk.relevance = 2
#         elif list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
#             blk.relevance = 1
#         else:
#             continue
#     ret = []
#
#     n0 = 1
#     pbuf_ht, nbuf_ht = dbuf.filtered(lambda blk, idx: blk.relevance >= 2, need_residue=True)
#     pbuf_b, nbuf_b = nbuf_ht.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
#
#     for i in range(n0):
#         _selected_htblks = random.sample(pbuf_ht.blocks, min(max_blk_num, len(pbuf_ht)))
#         _selected_pblks = random.sample(pbuf_b.blocks, min(max_blk_num - len(_selected_htblks), len(pbuf_b)))
#         _selected_nblks = random.sample(nbuf_b.blocks, min(max_blk_num - len(_selected_pblks) - len(_selected_htblks), len(nbuf_b)))
#         buf = Buffer()
#         buf.blocks = _selected_htblks + _selected_pblks + _selected_nblks
#         ret.append(buf.sort_())
#     ret[0][0].ids.insert(0, tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
#     return ret[0]

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

    if '/' in doc1_id:
        doc1_id = doc1_id.replace('/', '_')
    if '/' in doc2_id:
        doc2_id = doc2_id.replace('/', '_')

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
    if args.train:
        mod = 'train'
    elif args.dev:
        mod = 'dev'
    elif args.test:
        mod = 'test'

    os.makedirs(f'{senemb_path}/{mod}', exist_ok=True)

    json.dump(senemb1, open(f'{senemb_path}/{mod}/codred-doc-senemb-doc1-{h}-{t}-{doc1_id}-{doc2_id}', 'w'))
    json.dump(senemb2, open(f'{senemb_path}/{mod}/codred-doc-senemb-doc2-{h}-{t}-{doc1_id}-{doc2_id}', 'w'))

    return k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2

def collate_fn(lst_batch, args, relation2id, tokenizer, redisd, sent_model):
    # assert len(lst_batch) == 1
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
                k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2 = process_example(args, h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
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
            continue # save
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
    # assert len(batch) == 1
    batch = batch[0]
    h, t = batch[0].split('#')
    rs = [relation2id[r] for r in batch[1]]
    dps = batch[2]

    tokens_codre = list()
    intervals_codre = list()
    target_emb_codre = list()
    lst_emb_codre = list()

    for doc1, doc2, l in dps:
        k1, intervals1, target_emb1, lst_emb1, k2, intervals2, target_emb2, lst_emb2 = process_example(args, h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, sent_model)
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
        self.d_model = 768
        self.reduced_dim = 256
        self.aggregator = args.aggregator
        self.no_doc_pair_supervision = args.no_doc_pair_supervision
        if args.v2:
            self.v2 = True
            num_layer = 3
        else:
            self.v2 = False
            num_layer = 4
        # self.matt = MatTransformer(h = 8 , d_model = self.d_model , hidden_size = 1024 , num_layers = num_layer , device = 'cuda')
        self.wu = nn.Linear(self.d_model , self.d_model)
        self.wv = nn.Linear(self.d_model , self.d_model)
        self.wi = nn.Linear(self.d_model , self.d_model)
        self.ln1 = nn.Linear(self.d_model , self.d_model)
        self.gamma = 2
        self.alpha = 0.25
        self.beta = 0.01
        self.d_k = 64
        self.num_relations = num_relations


    def forward(self, lst_input_ids, lst_token_type_ids, lst_attention_mask, lst_dplabel=None, lst_rs=None, need_dplogit=False, train=True):
        f_loss = 0.
        f_prediction = []
        f_logit = []
        f_dp_logit = []
        f_ht_logits_flatten = []
        f_ht_fixed_low = []
        f_num_b = []
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
            bag_len, seq_len = input_ids.size()
            embedding, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
            p_embedding = embedding[:, 0, :]
            # if bag_len>8:
            #     print("bag_len:", bag_len)
            num_b = []

            if rs is not None or not train:
                entity_mask, entity_span_list = self.get_htb(input_ids)

                for dp in range(0,bag_len):
                    num_b.append(len(entity_span_list[dp][2]))

                h_embs = []
                t_embs = []
                b_embs = []
                dp_embs = []
                for dp in range(0,bag_len):
                    b_embs_dp = []
                    try:
                        h_span = entity_span_list[dp][0]
                        t_span = entity_span_list[dp][1]
                        b_span_chunks = entity_span_list[dp][2]
                        h_emb = torch.max(embedding[dp, h_span[0]:h_span[-1]+1], dim=0)[0]
                        t_emb = torch.max(embedding[dp, t_span[0]:t_span[-1]+1], dim=0)[0]
                        h_embs.append(h_emb)
                        t_embs.append(t_emb)
                        for b_span in b_span_chunks:
                            b_emb = torch.max(embedding[dp, b_span[0]:b_span[1]+1], dim=0)[0]
                            b_embs_dp.append(b_emb)
                        if bag_len >= 16:
                            if len(b_embs_dp) > 3:
                                b_embs_dp = random.choices(b_embs_dp, k=3)
                        if bag_len >= 14:
                            if len(b_embs_dp) > 4:
                                b_embs_dp = random.choices(b_embs_dp, k=4)
                        elif bag_len >= 10:
                            if len(b_embs_dp) > 5:
                                b_embs_dp = random.choices(b_embs_dp, k=5)
                        else:
                            if len(b_embs_dp) > 8:
                                b_embs_dp = random.choices(b_embs_dp, k=8)
                            else:
                                b_embs_dp = b_embs_dp
                        b_embs.append(b_embs_dp)
                        dp_embs.append(p_embedding[dp])
                    except IndexError as e:
                        print('input_ids',input_ids,input_ids.size())
                        print('embedding', embedding, embedding.size())
                        print('entity_span_list', entity_span_list, len(entity_span_list), len(entity_span_list[0]))
                        continue

                predict_logits,  logit, dp_logit = self.predict_logit(p_embedding, embedding, h_embs, t_embs, b_embs, rs, train)
                ht_logits = predict_logits
                bag_logit = torch.max(ht_logits.transpose(0, 1), dim=1)[0].unsqueeze(0)
                path_logit = ht_logits
            else:     # Inner doc
                entity_mask, entity_span_list = self.get_htb(input_ids)

                for dp in range(0,bag_len):
                    num_b.append(len(entity_span_list[dp][2]))

                path_logits = []
                ht_logits_flatten_list = []
                for dp in range(0,bag_len):
                    h_embs = []
                    t_embs = []
                    b_embs = []
                    # try:
                    h_span = entity_span_list[dp][0]
                    t_span = entity_span_list[dp][1]
                    b_span_chunks = entity_span_list[dp][2]
                    h_emb = torch.max(embedding[dp, h_span[0]:h_span[-1]+1], dim=0)[0]
                    t_emb = torch.max(embedding[dp, t_span[0]:t_span[-1]+1], dim=0)[0]
                    h_embs.append(h_emb)
                    t_embs.append(t_emb)
                    for b_span in b_span_chunks:
                        b_emb = torch.max(embedding[dp, b_span[0]:b_span[1]+1], dim=0)[0]
                        b_embs.append(b_emb)
                    h_index = [1 for _ in h_embs]
                    t_index = [2 for _ in t_embs]

                    predict_logits, logit, dp_logit = self.predict_logit(p_embedding, embedding, h_embs, t_embs, b_embs, rs=None)
                    # torch.Size([1, 277]) torch.Size([8, 277])

                    ht_logits = predict_logits[0][:len(h_index), len(h_index):len(h_index)+len(t_index)]
                    _ht_logits_flatten = ht_logits.reshape(1, -1, self.num_relations)

                    ht_logits = predict_logits[0][:len(h_index), len(h_index):len(h_index)+len(t_index)]
                    path_logits.append(ht_logits)
                    ht_logits_flatten_list.append(_ht_logits_flatten)
                try:
                    path_logit = torch.stack(path_logits).reshape(1, 1, -1, self.num_relations).squeeze(0).squeeze(0)
                except Exception as e:
                    print(e)
                    pdb.set_trace()


            if dplabel is not None and rs is None: #ok
                ht_logits_flatten = torch.stack(ht_logits_flatten_list).squeeze(1)
                if self.v2:
                    ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 0)[:, :, 0].unsqueeze(-1)
                else:
                    ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 8)[:, :, 0].unsqueeze(-1)
                y_true = torch.zeros_like(ht_logits_flatten)
                for idx, dpl in enumerate(dplabel):
                    try:
                        y_true[idx, :, dpl.item()] = 1
                    except:
                        pass
                bag_logit = path_logit
                loss = self._multilabel_categorical_crossentropy(ht_logits_flatten, y_true, ht_fixed_low + 2,
                                                                 ht_fixed_low)
            elif rs is not None:
                if self.no_doc_pair_supervision:
                    pass
                else:
                    ht_logits_flatten = ht_logits.unsqueeze(1)
                    y_true = torch.zeros_like(ht_logits_flatten)
                    if self.v2:
                        ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 0)[:, :, 0].unsqueeze(-1)
                    else:
                        ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 8)[:, :, 0].unsqueeze(-1)
                    for idx, dpl in enumerate(dplabel):
                        y_true[idx, :, dpl.item()] = torch.ones_like(y_true[idx, :, dpl.item()])
                    loss = self._multilabel_categorical_crossentropy(ht_logits_flatten, y_true, ht_fixed_low + 2,
                                                                     ht_fixed_low)
            else:
                ht_logits_flatten = ht_logits.unsqueeze(1)
                if self.v2:
                    ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 0)[:, :, 0].unsqueeze(-1)
                else:
                    ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 8)[:, :, 0].unsqueeze(-1)
                loss = None

            prediction = []
            if loss is not None:
                f_loss += loss
            f_prediction.append(prediction)
            f_logit.append(logit)
            f_dp_logit.append(dp_logit)
            f_ht_logits_flatten.append(ht_logits_flatten.transpose(0,1))
            f_ht_fixed_low.append((ht_fixed_low+2).transpose(0,1))
            f_num_b.append(num_b)
        f_loss /= len(lst_input_ids)
        if need_dplogit:
            # _,          f_prediction, f_logit, f_ht_logits_flatten, ht_threshold_flatten = outputs
            return f_loss, f_prediction, f_logit, f_dp_logit, f_ht_logits_flatten, f_ht_fixed_low, f_num_b
        else:
            return f_loss, f_prediction, f_logit

    def predict_logit(self, r_embedding, embedding, h_embs, t_embs, b_embs, rs=None, train=True):
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
        elif self.aggregator == 'ecrim_attention' and (rs is not None or not train):
            htb_index = []
            htb_embs = []
            htb_start = [0]
            htb_end = []
            for h_emb, t_emb, b_emb in zip(h_embs, t_embs, b_embs):
                htb_embs.extend([h_emb, t_emb])
                htb_index.extend([1, 2])
                htb_embs.extend(b_emb)
                htb_index.extend([3] * len(b_emb))
                htb_end.append(len(htb_index) - 1)
                htb_start.append(len(htb_index))

            try:
                htb_embs_t = torch.stack(htb_embs, dim=0).unsqueeze(0)
            except:
                print(htb_embs)
                print(h_embs)
                print(t_embs)
                print(b_embs)

            u = self.wu(htb_embs_t)
            v = self.wv(htb_embs_t)

            alpha = u.view(1, len(htb_index), 1, htb_embs_t.size()[-1]) + v.view(1, 1, len(htb_index),
                                                                                 htb_embs_t.size()[-1])  # wu*i + wv*j
            alpha = F.relu(alpha)

            rel_enco = F.relu(self.ln1(alpha))

            rel_mask = torch.ones(1, len(htb_index), len(htb_index)).to(embedding.device)
            rel_enco_m = self.matt(rel_enco, rel_mask)
            h_pos = []
            t_pos = []
            for i, e_type in enumerate(htb_index):
                if e_type == 1:
                    h_pos.append(i)
                elif e_type == 2:
                    t_pos.append(i)
                else:
                    continue
            assert len(h_pos) == len(t_pos)
            rel_enco_m_ht = []

            for i, j in zip(h_pos, t_pos):
                rel_enco_m_ht.append(rel_enco_m[0][i][j])
            t_feature_m = torch.stack(rel_enco_m_ht)

            predict_logits = self.predictor(t_feature_m)

            attention_score = torch.matmul(r_embedding, torch.t(weight))
            attention_weight = torch.nn.functional.softmax(attention_score, dim=0)
            embedding = torch.matmul(torch.transpose(attention_weight, 0, 1), r_embedding)
            logit = self.predictor(embedding)

            return predict_logits, torch.diag(logit).unsqueeze(0), self.predictor(r_embedding)


        elif self.aggregator == 'ecrim_attention' and rs is None:
            h_index = [1 for _ in h_embs]
            t_index = [2 for _ in t_embs]
            b_index = [3 for _ in b_embs]
            htb_index = []
            htb_embs = []
            for idx, embs in zip([h_index, t_index, b_index], [h_embs, t_embs, b_embs]):
                htb_index.extend(idx)
                htb_embs.extend(embs)
            rel_mask = torch.ones(1, len(htb_index), len(htb_index)).to(embedding.device)

            htb_embs_t = torch.stack(htb_embs, dim=0).unsqueeze(0)

            u = self.wu(htb_embs_t)
            v = self.wv(htb_embs_t)
            alpha = u.view(1, len(htb_index), 1, htb_embs_t.size()[-1]) + v.view(1, 1, len(htb_index),
                                                                                 htb_embs_t.size()[-1])
            alpha = F.relu(alpha)

            rel_enco = F.relu(self.ln1(alpha))

            rel_enco_m = self.matt(rel_enco, rel_mask)

            t_feature = rel_enco_m
            bs, es, es, d = rel_enco.size()

            attention_score = torch.matmul(r_embedding, torch.t(weight))
            attention_weight = torch.nn.functional.softmax(attention_score, dim=0)
            embedding = torch.matmul(torch.transpose(attention_weight, 0, 1), r_embedding)
            logit = self.predictor(embedding)
            # return logit, self.predictor(t_feature.reshape(bs,es,es,d)) ####################here

            predict_logits = self.predictor(t_feature.reshape(bs,es,es,d))
            return predict_logits, torch.diag(logit).unsqueeze(0), self.predictor(r_embedding)
        else:
            assert False

    def _multilabel_categorical_crossentropy(self, y_pred, y_true, cr_ceil, cr_low, ghm=True, r_dropout=True):
        # cr_low + 2 = cr_ceil
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12  
        y_pred_pos = y_pred - (1 - y_true) * 1e12  
        y_pred_neg = torch.cat([y_pred_neg, cr_ceil], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, -cr_low], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return ((neg_loss + pos_loss + cr_low.squeeze(-1) - cr_ceil.squeeze(-1))).mean()

    def graph_encode(self , ent_encode , rel_encode , ent_mask , rel_mask):
        bs , ne , d = ent_encode.size()
        ent_encode = ent_encode + self.ent_emb[0].view(1,1,d)
        rel_encode = rel_encode + self.ent_emb[1].view(1,1,1,d)
        rel_encode , ent_encode = self.graph_enc(rel_encode , ent_encode , rel_mask , ent_mask)
        return rel_encode


    def get_htb(self, input_ids):
        htb_mask_list = []
        htb_list_batch = []
        for pi in range(input_ids.size()[0]):
            #pdb.set_trace()
            tmp = torch.nonzero(input_ids[pi] - torch.full(([input_ids.size()[1]]), 1).to(input_ids.device))
            if tmp.size()[0] < input_ids.size()[0]:
                print(input_ids)
            h_starts = [i[0] for i in (input_ids[pi] == H_START_MARKER_ID).nonzero().detach().tolist()]
            h_ends = [i[0] for i in (input_ids[pi] == H_END_MARKER_ID).nonzero().detach().tolist()]
            t_starts = [i[0] for i in (input_ids[pi] == T_START_MARKER_ID).nonzero().detach().tolist()]
            t_ends = [i[0] for i in (input_ids[pi] == T_END_MARKER_ID).nonzero().detach().tolist()]
            if len(h_starts) == len(h_ends):
                h_start = h_starts[0]
                h_end = h_ends[0]
            else:
                for h_s in h_starts:
                    for h_e in h_ends:
                        if 0 < h_e - h_s < 20:
                            h_start = h_s
                            h_end = h_e
                            break

            if len(t_starts) == len(t_ends):
                t_start = t_starts[0]
                t_end = t_ends[0]
            else:
                for t_s in t_starts:
                    for t_e in t_ends:
                        if 0 < t_e - t_s < 20:
                            t_start = t_s
                            t_end = t_e
                            break
            try:
                if h_end - h_start <= 0 or t_end - t_start <= 0:
                    # pdb.set_trace()
                    if h_end - h_start <= 0:
                        for h_s in h_starts:
                            for h_e in h_ends:
                                if 0 < h_e - h_s < 20:
                                    h_start = h_s
                                    h_end = h_e
                                    break
                    if t_end - t_start <= 0:
                        for t_s in t_starts:
                            for t_e in t_ends:
                                if 0 < t_e - t_s < 20:
                                    t_start = t_s
                                    t_end = t_e
                                    break
                    if h_end - h_start <= 0 or t_end - t_start <= 0:
                        pdb.set_trace()
            except:
                print(h_starts)
                print(h_ends)
                print(t_starts)
                print(t_ends)
                tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
                token = tokenizer.convert_ids_to_tokens(input_ids[pi])
                print('token',token)
            b_spans = torch.nonzero(
                torch.gt(torch.full(([input_ids.size()[1]]), 99).to(input_ids.device), input_ids[pi])).squeeze(
                0).squeeze(1).detach().tolist()
            token_len = input_ids[pi].nonzero().size()[0]
            b_spans = [i for i in b_spans if i <= token_len - 1]
            assert len(b_spans) >= 4
            # for i in [h_start, h_end, t_start, t_end]:
            for i in h_starts + h_ends + t_starts + t_ends:
                b_spans.remove(i)
            h_span = [h_pos for h_pos in range(h_start, h_end + 1)]
            t_span = [t_pos for t_pos in range(t_start, t_end + 1)]
            h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(h_span).to(
                input_ids.device), 1)
            t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(t_span).to(
                input_ids.device), 1)

            b_span_ = []
            if len(b_spans) > 0 and len(b_spans)%2==0:
                b_span_chunks = [b_spans[i:i+2] for i in range(0,len(b_spans),2)]
                b_span = []
                for span in b_span_chunks:
                    b_span.extend([b_pos for b_pos in range(span[0], span[1]+1)])
                b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(b_span).to(input_ids.device), 1)
                b_span_.extend(b_span)
            elif len(b_spans) > 0 and len(b_spans)%2==1:
                b_span = []
                ptr = 0
                #pdb.set_trace()
                while(ptr<=len(b_spans)-1):
                    try:
                        if input_ids[pi][b_spans[ptr+1]] - input_ids[pi][b_spans[ptr]] == 1:
                            b_span.append([b_spans[ptr], b_spans[ptr+1]])
                            ptr += 2
                        else:
                            ptr += 1
                    except IndexError as e:
                        ptr += 1 
                for bs in b_span:
                    b_span_.extend(bs)
                    if len(b_span_)%2 != 0:
                        print(b_spans)
                b_span_chunks = [b_span_[i:i+2] for i in range(0,len(b_span_),2)]
                b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(b_span_).to(input_ids.device), 1)
            else:
                b_span_ = []
                b_span_chunks = []
                b_mask = torch.zeros_like(input_ids[pi])
            htb_mask = torch.concat([h_mask.unsqueeze(0), t_mask.unsqueeze(0), b_mask.unsqueeze(0)], dim=0)
            htb_mask_list.append(htb_mask)
            htb_list_batch.append([h_span, t_span, b_span_chunks])
        htb_mask_batch = torch.stack(htb_mask_list,dim=0)
        return htb_mask_batch, htb_list_batch 

class CodredCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_argument(self, parser):
        parser.add_argument('--seq_len', type=int, default=512)
        parser.add_argument('--aggregator', type=str, default='ecrim_attention')
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
        parser.add_argument('--model_name', type=str, default='bert')

    def load_model(self):
        relations = json.load(open('../../../data/rawdata/relations.json'))
        relations.sort()
        self.relations = ['n/a'] + relations
        self.relation2id = dict()
        for index, relation in enumerate(self.relations):
            self.relation2id[relation] = index
        with self.trainer.cache():
            model = Codred(self.args, len(self.relations))
            selector = LSTMSelector(768, 512, self.args.lstm_size, self.args.epsilon)
            if self.args.load_model_path:
                state_dict = torch.load(self.args.load_model_path, map_location=torch.device('cpu'))
                state_keys = list(state_dict.keys())
                for key in state_keys:
                    if key.startswith('module.'):
                        v = state_dict[key]
                        del state_dict[key]
                        state_dict[key[7:]] = v
                model.load_state_dict(state_dict, strict=False)
            if self.args.load_selector_path:
                load_model(selector, self.args.load_selector_path)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.tokenizer = tokenizer

        self.sent_model = BertModel.from_pretrained('bert-base-cased').cuda()
        self.sent_model.eval()

        # self.sbert_wk = sbert(device='cuda')
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
        return partial(collate_fn, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model),\
               partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model),\
               partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, sent_model=self.sent_model)

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        with self.trainer.once():
            self.train_logger.log(train_loss=loss)
            _, f_prediction, f_logit, f_dplogit, f_ht_logits_flatten, f_ht_threshold_flatten, f_num_b = outputs
            for i in range(len(f_logit)):
                try:
                    if inputs['lst_rs'][i] is not None:
                        logit = f_logit[i]
                        ht_logits_flatten = f_ht_logits_flatten[i]
                        ht_threshold_flatten = f_ht_threshold_flatten[i]
                        rs = extra['lst_rs'][i]

                        if ht_logits_flatten is not None:
                            r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)
                            if r_score > ht_threshold_flatten[0, 0, 0]:
                                prediction = [r_idx.item()]
                            else:
                                prediction = [0]

                        logit = tensor_to_obj(logit)
                        for p, score, gold in zip(prediction, logit, rs):
                            self.train_logger.log(train_acc=1 if p == gold else 0)
                            if gold > 0:
                                self.train_logger.log(train_pos_acc=1 if p == gold else 0)

                    else:
                        logit = f_logit[i]
                        ht_logits_flatten = f_ht_logits_flatten[i]
                        ht_threshold_flatten = f_ht_threshold_flatten[i]
                        dplabel = inputs['lst_dplabel'][i]
                        logit, dplabel = tensor_to_obj(logit, dplabel)
                        prediction = []
                        if ht_logits_flatten is not None:
                            r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)
                            for dp_i, (r_s, r_i) in enumerate(zip(r_score, r_idx)):
                                if r_s > ht_threshold_flatten[dp_i, 0, 0]:
                                    prediction.append(r_i.item())
                                else:
                                    prediction.append(0)

                        for p, l in zip(prediction, dplabel):
                            self.train_logger.log(train_dsre_acc=1 if p == l else 0)
                except:
                    print("List Index Size")
                    print(len(f_prediction),len(f_logit), len(f_dplogit), len(f_ht_logits_flatten), len(f_ht_threshold_flatten))
                    print(len(inputs['lst_rs']))

    def on_train_epoch_end(self, epoch):
        print(epoch, self.train_logger.d)
        pass

    def on_dev_epoch_start(self, epoch):
        self._prediction = list()

    def on_dev_step(self, step, inputs, extra, outputs):
        # logit [1, 277]
        # ht_logits_flatten [1, 8, 277]
        # ht_threshold_flatten [1, 8, 1]
        _, f_prediction, f_logit, f_dplogit, f_ht_logits_flatten, f_ht_threshold_flatten, f_num_b = outputs
        h, t, rs = extra['h'], extra['t'], extra['rs']
        for i in range(len(f_prediction)):
            #ecrim
            ht_logits_flatten = f_ht_logits_flatten[i]
            ht_threshold_flatten = f_ht_threshold_flatten[i]
            r_score, r_idx = torch.max(torch.max(ht_logits_flatten,dim=1)[0], dim=-1)
            eval_logit = torch.max(ht_logits_flatten,dim=1)[0]
            if any(r_score>ht_threshold_flatten[:, 0, 0]):
                prediction = [r_idx.item()]
            else:
                prediction = [0]
            self._prediction.append([prediction[0], eval_logit[0], h, t, rs])

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
            json.dump(stat, open(f'output/dev-stat-dual-K1-{epoch}.json', 'w'))
            json.dump(results, open(f'output/dev-results-dual-K1-{epoch}.json', 'w'))
        return stat['f1']

    def on_test_epoch_start(self, epoch):
        self._prediction = list()
        pass

    def on_test_step(self, step, inputs, extra, outputs):
        _, f_prediction, f_logit, f_dplogit, f_ht_logits_flatten, f_ht_threshold_flatten, f_num_b = outputs
        h, t, rs = extra['h'], extra['t'], extra['rs']
        for i in range(len(f_prediction)):
            ht_logits_flatten = f_ht_logits_flatten[i]
            ht_threshold_flatten = f_ht_threshold_flatten[i]
            r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)
            eval_logit = torch.max(ht_logits_flatten, dim=1)[0]
            if any(r_score > ht_threshold_flatten[:, 0, 0]):
                prediction = [r_idx.item()]
            else:
                prediction = [0]
            self._prediction.append([prediction[0], eval_logit[0], h, t, rs])

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
        return True

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
        inputs_codre = {
            'lst_tokens_codre': data[0],
            'lst_intervals_codre': data[1],
            'lst_target_emb_codre': data[2],
            'lst_lst_emb_codre': data[3]
        }
        return inputs_codre, {'h': data[4], 'rs': data[5], 't': data[6]}

    def process_test_data(self, data):
        inputs_codre = {
            'lst_tokens_codre': data[0],
            'lst_intervals_codre': data[1],
            'lst_target_emb_codre': data[2],
            'lst_lst_emb_codre': data[3]
        }
        return inputs_codre, {'h': data[4], 'rs': data[5], 't': data[6]}

def main():
    trainer = Trainer(CodredCallback())
    trainer.run()


if __name__ == '__main__':
    main()
