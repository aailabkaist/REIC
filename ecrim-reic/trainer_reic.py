import argparse
import json
import logging
import os
import random
import shutil
import sys
import pdb
import apex
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
# from data_helper import BlkPosInterface, SimpleListDataset
import pickle
import torch.nn as nn


class ContextError(Exception):
    def __init__(self):
        pass


class Once:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank > 0:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        return True

    def trace(self, frame, event, arg):
        raise ContextError

    def __exit__(self, type, value, traceback):
        if type == ContextError:
            return True
        else:
            return False


class OnceBarrier:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank > 0:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        return True

    def trace(self, frame, event, arg):
        raise ContextError

    def __exit__(self, type, value, traceback):
        if self.rank >= 0:
            torch.distributed.barrier()
        if type == ContextError:
            return True
        else:
            return False


class Cache:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank not in [-1, 0]:
            torch.distributed.barrier()
        return True

    def __exit__(self, type, value, traceback):
        if self.rank == 0:
            torch.distributed.barrier()
        return False


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class Prefetcher:
    def __init__(self, dataloader, stream):
        self.dataloader = dataloader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        try:
            self.next = next(self.iter)
        except StopIteration:
            self.next = None
            return
        with torch.cuda.stream(self.stream):
            next_list = list()
            for v in self.next:
                if type(v) == torch.Tensor:
                    next_list.append(v.cuda(non_blocking=True))
                else:
                    next_list.append(v)
            self.next = tuple(next_list)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next is not None:
            result = self.next
            self.preload()
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)


class TrainerCallback:
    def __init__(self):
        pass

    def on_argument(self, parser):
        pass

    def load_model(self):
        pass

    def load_data(self):
        pass

    def collate_fn(self):
        return None, None, None

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_dev_epoch_start(self, epoch):
        pass

    def on_dev_step(self, step, inputs, extra, outputs):
        pass

    def on_dev_epoch_end(self, epoch):
        pass

    def on_test_epoch_start(self, epoch):
        pass

    def on_test_step(self, step, inputs, extra, outputs):
        pass

    def on_test_epoch_end(self, epoch):
        pass

    def process_train_data(self, data):
        pass

    def process_dev_data(self, data):
        pass

    def process_test_data(self, data):
        pass


    def on_save(self, path):
        pass

    def on_load(self, path):
        pass


class Trainer:
    def __init__(self, callback: TrainerCallback):
        self.callback = callback
        self.callback.trainer = self
        logging.basicConfig(level=logging.INFO)

    def parse_args(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--dev', action='store_true')
        self.parser.add_argument('--test', action='store_true')

        self.parser.add_argument('--senemb_path', default='../data/senemb' ,type=str)


        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument("--per_gpu_train_batch_size", default=1, type=int)
        self.parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int)
        self.parser.add_argument("--learning_rate", default=5e-5, type=float)
        self.parser.add_argument("--selector_learning_rate", default=3e-3, type=float)
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        self.parser.add_argument("--weight_decay", default=0.0, type=float)
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float)
        self.parser.add_argument("--epochs", default=10, type=int)
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
        self.parser.add_argument("--intro_save", default="../data/", type=str)
        self.parser.add_argument("--num_sentences", default=3, type=int)

        self.parser.add_argument("--lam_notnone", default=10, type=float)
        self.parser.add_argument("--lam_bridge", default=0, type=float)
        self.parser.add_argument("--lam_none", default=1, type=float)
        self.parser.add_argument("--lstm_size", default=512, type=int)
        self.parser.add_argument("--reward_limit", default=9999, type=float)
        self.parser.add_argument("--epsilon", default=0, type=float)

        self.parser.add_argument("--v2", action="store_true")


        # self.parser.add_argument("--model_name", default="bert", type=str)
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
        self.model_name = self.args.model_name
        self.intro_save = self.args.intro_save
        self.reward_limit = self.args.reward_limit


    def set_env(self):
        if self.debug:
            sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1
        set_seed(self.seed, self.n_gpu)
        self.device = device
        with self.once_barrier():
            if not os.path.exists('r'):
                os.mkdir('r')
            runs = os.listdir('r')
            i = max([int(c) for c in runs], default=-1) + 1
            os.mkdir(os.path.join('r', str(i)))
            src_names = [source for source in os.listdir() if source.endswith('.py')]
            for src_name in src_names:
                shutil.copy(src_name, os.path.join('r', str(i), src_name))
            os.mkdir(os.path.join('r', str(i), 'output'))
            os.mkdir(os.path.join('r', str(i), 'tmp'))
        runs = os.listdir('r')
        i = max([int(c) for c in runs])
        os.chdir(os.path.join('r', str(i)))
        with self.once_barrier():
            json.dump(sys.argv, open('output/args.json', 'w'))
        logging.info("Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.local_rank, device, self.n_gpu, bool(self.local_rank != -1), self.fp16))
        # self.train_batch_size = self.per_gpu_train_batch_size
        # self.eval_batch_size = self.per_gpu_eval_batch_size
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)

        if self.fp16:
            apex.amp.register_half_function(torch, "einsum")
        self.stream = torch.cuda.Stream()

    def set_model(self):
        self.model, self.selector = self.callback.load_model()
        self.model.to(self.device)
        self.selector.to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.selector_optimizer = torch.optim.AdamW(self.selector.parameters(), lr=self.selector_learning_rate, eps=self.adam_epsilon)

        if self.fp16:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)
            self.selector, self.selector_optimizer = apex.amp.initialize(self.selector, self.selector_optimizer, opt_level=self.fp16_opt_level)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.selector = torch.nn.DataParallel(self.selector)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=True)
            self.selector = torch.nn.parallel.DistributedDataParallel(self.selector, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def once(self):
        return Once(self.local_rank)

    def once_barrier(self):
        return OnceBarrier(self.local_rank)

    def cache(self):
        return Cache(self.local_rank)

    def load_data(self):
        self.train_step = 1
        self.epochs_trained = 0
        self.steps_trained_in_current_epoch = 0
        self.intro_train_step = 1
        train_dataset, dev_dataset, test_dataset = self.callback.load_data()
        # train_dataset, dev_dataset = self.callback.load_data()
        train_fn, dev_fn, test_fn = self.callback.collate_fn()
        if train_dataset:
            if self.dataset_ratio < 1:
                train_dataset = torch.utils.data.Subset(train_dataset,
                                                        list(range(int(len(train_dataset) * self.dataset_ratio))))
            self.train_dataset = train_dataset
            self.train_sampler = RandomSampler(self.train_dataset) if self.local_rank == -1 else DistributedSampler(
                self.train_dataset)
            self.train_dataloader = Prefetcher(
                DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=self.train_batch_size,
                           collate_fn=train_fn, num_workers=self.num_workers), self.stream)
            self.t_total = len(self.train_dataloader) // self.gradient_accumulation_steps * self.epochs
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=int(self.t_total * self.warmup_ratio),
                                                             num_training_steps=self.t_total)
        if dev_dataset:
            if self.dataset_ratio < 1:
                dev_dataset = torch.utils.data.Subset(dev_dataset,
                                                      list(range(int(len(dev_dataset) * self.dataset_ratio))))
            self.dev_dataset = dev_dataset
            self.dev_sampler = SequentialSampler(self.dev_dataset) if self.local_rank == -1 else DistributedSampler(
                self.dev_dataset)
            self.dev_dataloader = Prefetcher(
                DataLoader(self.dev_dataset, sampler=self.dev_sampler, batch_size=self.eval_batch_size,
                           collate_fn=dev_fn, num_workers=self.num_workers), self.stream)
        if test_dataset:
            if self.dataset_ratio < 1:
                test_dataset = torch.utils.data.Subset(test_dataset,
                                                       list(range(int(len(test_dataset) * self.dataset_ratio))))
            self.test_dataset = test_dataset
            self.test_sampler = SequentialSampler(self.test_dataset) if self.local_rank == -1 else DistributedSampler(
                self.test_dataset)
            self.test_dataloader = Prefetcher(
                DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=self.eval_batch_size,
                           collate_fn=test_fn, num_workers=self.num_workers), self.stream)

    def restore_checkpoint(self, path, ignore_progress=False):
        if self.no_save:
            return
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=self.device))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt"), map_location=self.device))
        self.scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt"), map_location=self.device))

        selector_to_load = self.selector.module if hasattr(self.model, "module") else self.selector
        selector_to_load.load_state_dict(torch.load(os.path.join(path, 'selector_pytorch_model.bin'), map_location=self.device))
        self.selector_optimizer.load_state_dict(torch.load(os.path.join(path, "selector_optimizer.pt"), map_location=self.device))

        self.callback.on_load(path)
        if not ignore_progress:
            self.train_step = int(path.split("-")[-1])
            self.epochs_trained = self.train_step // (len(self.train_dataloader) // self.gradient_accumulation_steps)
            self.steps_trained_in_current_epoch = self.train_step % (
                        len(self.train_dataloader) // self.gradient_accumulation_steps)
        logging.info("  Continuing training from checkpoint, will skip to saved train_step")
        logging.info("  Continuing training from epoch %d", self.epochs_trained)
        logging.info("  Continuing training from train step %d", self.train_step)
        logging.info("  Will skip the first %d steps in the first epoch", self.steps_trained_in_current_epoch)

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
        torch.save(self.selector_scheduler.state_dict(), os.path.join(output_dir, "selector_scheduler.pt"))

        self.callback.on_save(output_dir)

    def check_unused(self, token):
        for kkk in reversed(range(len(token))):
            if '[unused' in token[kkk]:
                if len(token[kkk]) == 9:
                    target = int(token[kkk][7])
                elif len(token[kkk]) == 10:
                    target = int(token[kkk][7:9])
                else:
                    import pdb
                    pdb.set_trace()
                if target > 4:
                    if target % 2 == 0:
                        if f'[unused{target - 1}]' not in token:
                            token = token[:kkk] + token[kkk + 1:]
                    else:
                        if f'[unused{target + 1}]' not in token:
                            token = token + [f'[unused{target + 1}]']
        return token

    def run(self):
        self.parse_args()
        self.set_env()
        with self.once():
            self.writer = torch.utils.tensorboard.SummaryWriter()
        self.set_model()
        self.load_data()
        self.selector_scheduler = get_linear_schedule_with_warmup(self.selector_optimizer,
                                                         num_warmup_steps=int(self.t_total * self.warmup_ratio),
                                                         num_training_steps=self.t_total)
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
                print("==========Training==========")
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

                            selection = torch.stack(selection).cpu().numpy()
                            selection.sort(axis=0)

                            log_probs = log_probs.reshape(-1, 2).sum(dim=1)
                            lst_log_probs.append(log_probs)

                            for idx, (token, interval) in enumerate(zip(tokens, intervals)):
                                target_st = 0
                                target_en = -1
                                for l in range(len(interval[0])):
                                    if target_st != interval[0][l][0]:
                                        target_en = interval[0][l][0]
                                        try:
                                            for j in range(10): # entity 내에 .이나 ,가 있는 경우
                                                if token[0][target_en + j] == '[unused2]':
                                                    interval[0][l][0] += j+1
                                                    target_en += j+1
                                                    break
                                        except:
                                            pass
                                        break
                                    else:
                                        target_st = interval[0][l][1]
                                if target_en == -1:
                                    target_en = len(token[0])

                                target_k1 = self.check_unused(token[0][target_st:target_en])
                                k1 = []

                                insert_target = False
                                loc_target_st = -1
                                loc_target_en = -1

                                for k in range(min(num_selection, len(interval[0]))):
                                    tmp_sel = selection[k][2*idx]
                                    st = interval[0][tmp_sel][0]
                                    en = interval[0][tmp_sel][1]

                                    if not insert_target:
                                        if target_en <= st:
                                            loc_target_st = len(k1)
                                            k1 = k1 + target_k1
                                            insert_target = True
                                            loc_target_en = len(k1)

                                    tmp_k1 = self.check_unused(token[0][st:en])
                                    k1 = k1 + tmp_k1

                                if not insert_target:
                                    loc_target_st = len(k1)
                                    k1 = k1 + target_k1
                                    loc_target_en = len(k1)

                                if len(k1) > BERT_MAX_LEN // 2 - 2:
                                    len_target = loc_target_en - loc_target_st
                                    len_other = BERT_MAX_LEN // 2 - 2 - len_target
                                    len_left = len_other // 2
                                    len_right = len_other - len_left

                                    left_margin = loc_target_st
                                    right_margin = len(k1) - loc_target_en

                                    if len_left > left_margin:
                                        len_right += len_left - left_margin
                                        len_left = left_margin
                                    if len_right > right_margin:
                                        len_left += len_right - right_margin
                                        len_right = right_margin

                                    k1 = k1[loc_target_st-len_left:loc_target_en+len_right]

                                target_st = 0
                                target_en = -1

                                for l in range(len(interval[1])):
                                    if target_st != interval[1][l][0]:
                                        target_en = interval[1][l][0]
                                        try:
                                            for j in range(10):
                                                if token[1][target_en + j] == '[unused4]':
                                                    interval[1][l][0] += j+1
                                                    target_en += j+1
                                                    break
                                        except:
                                            pass
                                        break
                                    else:
                                        target_st = interval[1][l][1]
                                if target_en == -1:
                                    target_en = len(token[1])

                                target_k2 = self.check_unused(token[1][target_st:target_en])
                                k2 = []

                                insert_target = False
                                loc_target_st = -1
                                loc_target_en = -1

                                for k in range(min(num_selection, len(interval[1]))):
                                    tmp_sel = selection[k][2 * idx + 1]
                                    st = interval[1][tmp_sel][0]
                                    en = interval[1][tmp_sel][1]

                                    if not insert_target:
                                        if target_en <= st:
                                            loc_target_st = len(k2)
                                            k2 = k2 + target_k2
                                            insert_target = True
                                            loc_target_en = len(k2)

                                    tmp_k2 = self.check_unused(token[1][st:en])
                                    k2 = k2 + tmp_k2

                                if not insert_target:
                                    loc_target_st = len(k2)
                                    k2 = k2 + target_k2
                                    loc_target_en = len(k2)

                                if len(k2) > BERT_MAX_LEN // 2 - 1:
                                    len_target = loc_target_en - loc_target_st
                                    len_other = BERT_MAX_LEN // 2 - 1 - len_target
                                    len_left = len_other // 2
                                    len_right = len_other - len_left

                                    left_margin = loc_target_st
                                    right_margin = len(k2) - loc_target_en

                                    if len_left > left_margin:
                                        len_right += len_left - left_margin
                                        len_left = left_margin
                                    if len_right > right_margin:
                                        len_left += len_right - right_margin
                                        len_right = right_margin

                                    k2 = k2[loc_target_st-len_left:loc_target_en+len_right]

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
                        lst_reward = []
                        for idx, log_probs in enumerate(lst_log_probs):
                            pred_prob = outputs[4][num_dsre + idx][0].detach()
                            thr = outputs[5][num_dsre + idx][0]
                            num_b = outputs[6][num_dsre + idx]
                            dplabel = inputs['lst_dplabel'][num_dsre + idx]
                            for idx_path, log_prob in enumerate(log_probs):
                                ######################
                                reward = pred_prob[idx_path][dplabel[idx_path]] - thr[idx_path][0]
                                reward = reward + num_b[idx_path] * self.args.lam_bridge
                                lst_reward.append(reward)
                                ######################
                        k = 0
                        reward_mean = torch.mean(torch.stack(lst_reward))
                        for idx, log_probs in enumerate(lst_log_probs):
                            dplabel = inputs['lst_dplabel'][num_dsre + idx]
                            for idx_path, log_prob in enumerate(log_probs):
                                reward = lst_reward[k] - reward_mean
                                if dplabel[idx_path] != 0:
                                    reward = reward * self.args.lam_notnone
                                else:
                                    reward = min(reward, self.args.reward_limit)
                                    reward = reward * self.args.lam_none
                                s_loss = s_loss - reward * log_prob
                                k = k + 1

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
                            self.selector_scheduler.step()
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
                    ##############
                    # print("HERE HERE HERE HERE HERE")
                    # print(inputs)
                    # print(outputs)
                    ##############
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

                        selection = torch.stack(selection).cpu().numpy()
                        selection.sort(axis=0)

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

                            target_k1 = self.check_unused(token[0][target_st:target_en])
                            k1 = []

                            insert_target = False
                            loc_target_st = -1
                            loc_target_en = -1

                            for k in range(min(num_selection, len(interval[0]))):
                                tmp_sel = selection[k][2*idx]
                                st = interval[0][tmp_sel][0]
                                en = interval[0][tmp_sel][1]

                                if not insert_target:
                                    if target_en <= st:
                                        loc_target_st = len(k1)
                                        k1 = k1 + target_k1
                                        insert_target = True
                                        loc_target_en = len(k1)

                                tmp_k1 = self.check_unused(token[0][st:en])
                                k1 = k1 + tmp_k1

                            if not insert_target:
                                loc_target_st = len(k1)
                                k1 = k1 + target_k1
                                loc_target_en = len(k1)

                            if len(k1) > BERT_MAX_LEN // 2 - 2:
                                len_target = loc_target_en - loc_target_st
                                len_other = BERT_MAX_LEN // 2 - 2 - len_target
                                len_left = len_other // 2
                                len_right = len_other - len_left

                                left_margin = loc_target_st
                                right_margin = len(k1) - loc_target_en

                                if len_left > left_margin:
                                    len_right += len_left - left_margin
                                    len_left = left_margin
                                if len_right > right_margin:
                                    len_left += len_right - right_margin
                                    len_right = right_margin

                                k1 = k1[loc_target_st - len_left:loc_target_en + len_right]

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

                            target_k2 = self.check_unused(token[1][target_st:target_en])
                            k2 = []

                            insert_target = False
                            loc_target_st = -1
                            loc_target_en = -1

                            for k in range(min(num_selection, len(interval[1]))):
                                tmp_sel = selection[k][2 * idx + 1]
                                st = interval[1][tmp_sel][0]
                                en = interval[1][tmp_sel][1]

                                if not insert_target:
                                    if target_en <= st:
                                        loc_target_st = len(k2)
                                        k2 = k2 + target_k2
                                        insert_target = True
                                        loc_target_en = len(k2)

                                tmp_k2 = self.check_unused(token[1][st:en])
                                k2 = k2 + tmp_k2

                            if not insert_target:
                                loc_target_st = len(k2)
                                k2 = k2 + target_k2
                                loc_target_en = len(k2)

                            if len(k2) > BERT_MAX_LEN // 2 - 1:
                                len_target = loc_target_en - loc_target_st
                                len_other = BERT_MAX_LEN // 2 - 1 - len_target
                                len_left = len_other // 2
                                len_right = len_other - len_left

                                left_margin = loc_target_st
                                right_margin = len(k2) - loc_target_en

                                if len_left > left_margin:
                                    len_right += len_left - left_margin
                                    len_left = left_margin
                                if len_right > right_margin:
                                    len_left += len_right - right_margin
                                    len_right = right_margin

                                k2 = k2[loc_target_st - len_left:loc_target_en + len_right]

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
                            'lst_attention_mask': attention_mask_t,
                        }

                        outputs = self.model(**inputs, need_dplogit=True, train=False)
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
                    inputs_codre, extra = self.callback.process_test_data(batch)

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

                    selection = torch.stack(selection).cpu().numpy()
                    selection.sort(axis=0)

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

                        target_k1 = self.check_unused(token[0][target_st:target_en])
                        k1 = []

                        insert_target = False
                        loc_target_st = -1
                        loc_target_en = -1

                        for k in range(min(num_selection, len(interval[0]))):
                            tmp_sel = selection[k][2 * idx]
                            st = interval[0][tmp_sel][0]
                            en = interval[0][tmp_sel][1]

                            if not insert_target:
                                if target_en <= st:
                                    loc_target_st = len(k1)
                                    k1 = k1 + target_k1
                                    insert_target = True
                                    loc_target_en = len(k1)

                            tmp_k1 = self.check_unused(token[0][st:en])
                            k1 = k1 + tmp_k1

                        if not insert_target:
                            loc_target_st = len(k1)
                            k1 = k1 + target_k1
                            loc_target_en = len(k1)

                        if len(k1) > BERT_MAX_LEN // 2 - 2:
                            len_target = loc_target_en - loc_target_st
                            len_other = BERT_MAX_LEN // 2 - 2 - len_target
                            len_left = len_other // 2
                            len_right = len_other - len_left

                            left_margin = loc_target_st
                            right_margin = len(k1) - loc_target_en

                            if len_left > left_margin:
                                len_right += len_left - left_margin
                                len_left = left_margin
                            if len_right > right_margin:
                                len_left += len_right - right_margin
                                len_right = right_margin

                            k1 = k1[loc_target_st - len_left:loc_target_en + len_right]

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

                        target_k2 = self.check_unused(token[1][target_st:target_en])
                        k2 = []

                        insert_target = False
                        loc_target_st = -1
                        loc_target_en = -1

                        for k in range(min(num_selection, len(interval[1]))):
                            tmp_sel = selection[k][2 * idx + 1]
                            st = interval[1][tmp_sel][0]
                            en = interval[1][tmp_sel][1]

                            if not insert_target:
                                if target_en <= st:
                                    loc_target_st = len(k2)
                                    k2 = k2 + target_k2
                                    insert_target = True
                                    loc_target_en = len(k2)

                            tmp_k2 = self.check_unused(token[1][st:en])
                            k2 = k2 + tmp_k2

                        if not insert_target:
                            loc_target_st = len(k2)
                            k2 = k2 + target_k2
                            loc_target_en = len(k2)

                        if len(k2) > BERT_MAX_LEN // 2 - 1:
                            len_target = loc_target_en - loc_target_st
                            len_other = BERT_MAX_LEN // 2 - 1 - len_target
                            len_left = len_other // 2
                            len_right = len_other - len_left

                            left_margin = loc_target_st
                            right_margin = len(k2) - loc_target_en

                            if len_left > left_margin:
                                len_right += len_left - left_margin
                                len_left = left_margin
                            if len_right > right_margin:
                                len_left += len_right - right_margin
                                len_right = right_margin

                            k2 = k2[loc_target_st - len_left:loc_target_en + len_right]

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
                        'lst_attention_mask': attention_mask_t,
                    }

                    outputs = self.model(**inputs, need_dplogit=True, train=False)
                    self.callback.on_test_step(step, inputs, extra, outputs)
                self.callback.on_test_epoch_end(epoch)
        with self.once():
            self.writer.close()
        json.dump(True, open('output/f.json', 'w'))

    def distributed_broadcast(self, l):
        assert type(l) == list or type(l) == dict
        if self.local_rank < 0:
            return l
        else:
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            json.dump(l, open(f'tmp/{self.local_rank}.json', 'w'))
            torch.distributed.barrier()
            objs = list()
            for i in range(process_number):
                objs.append(json.load(open(f'tmp/{i}.json')))
            if type(objs[0]) == list:
                ret = list()
                for i in range(process_number):
                    ret.extend(objs[i])
            else:
                ret = dict()
                for i in range(process_number):
                    for k, v in objs.items():
                        assert k not in ret
                        ret[k] = v
            torch.distributed.barrier()
            return ret

    def distributed_merge(self, l):
        assert type(l) == list or type(l) == dict
        if self.local_rank < 0:
            return l
        else:
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            json.dump(l, open(f'tmp/{self.local_rank}.json', 'w'))
            torch.distributed.barrier()
            if self.local_rank == 0:
                objs = list()
                for i in range(process_number):
                    objs.append(json.load(open(f'tmp/{i}.json')))
                if type(objs[0]) == list:
                    ret = list()
                    for i in range(process_number):
                        ret.extend(objs[i])
                else:
                    ret = dict()
                    for i in range(process_number):
                        for k, v in objs.items():
                            assert k not in ret
                            ret[k] = v
            else:
                ret = None
            torch.distributed.barrier()
            return ret

    def distributed_get(self, v):
        if self.local_rank < 0:
            return v
        else:
            torch.distributed.barrier()
            if self.local_rank == 0:
                json.dump(v, open('tmp/v.json', 'w'))
            torch.distributed.barrier()
            v = json.load(open('tmp/v.json'))
            torch.distributed.barrier()
            return v

    def _write_estimation(self, buf, relevance_blk, f):
        for i, blk in enumerate(buf):
            f.write(f'{blk.pos} {relevance_blk[i].item()}\n')

    def _score_blocks(self, qbuf, relevance_token):
        ends = qbuf.block_ends()
        relevance_blk = torch.ones(len(ends), device='cpu')
        for i in range(len(ends)):
            if qbuf[i].blk_type > 0:  # query
                relevance_blk[i] = (relevance_token[ends[i - 1]:ends[i]]).mean()
        return relevance_blk

    def _collect_estimations_from_dir(self, est_dir):
        ret = {}
        for shortname in os.listdir(est_dir):
            filename = os.path.join(est_dir, shortname)
            if shortname.startswith('estimations_'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        ret[pos].estimation = estimation
                os.replace(filename, os.path.join(est_dir, 'backup_' + shortname))
        return ret

class LSTMSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_dim, epsilon, fn_activate='tanh'):
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

        self.epsilon = epsilon


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
                if np.random.rand() < self.epsilon:
                    probs = torch.softmax(torch.ones_like(sc) + mask, dim=-1)  # bs x ns
                else:
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

