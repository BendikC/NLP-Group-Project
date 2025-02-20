# Standard library imports
import sys
import os
import math
import json
import torch
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List

logger = logging.getLogger(__name__)


class TextTokenIdsCache:
    """Cache for storing and retrieving tokenized text sequences from memory-mapped files"""
    def __init__(self, data_dir, prefix):
        # Load metadata containing shape information
        meta = json.load(open(f"{data_dir}/{prefix}_meta"))
        self.total_number = meta['total_number']  # Total number of sequences
        self.max_seq_len = meta['embedding_size']  # Maximum sequence length
        
        try:
            # Try loading from root data directory
            self.ids_arr = np.memmap(f"{data_dir}/{prefix}.memmap", 
                shape=(self.total_number, self.max_seq_len), 
                dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/{prefix}_length.npy")
        except FileNotFoundError:
            # Fallback to memmap subdirectory
            self.ids_arr = np.memmap(f"{data_dir}/memmap/{prefix}.memmap", 
                shape=(self.total_number, self.max_seq_len), 
                dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/memmap/{prefix}_length.npy")
            
        # Verify lengths array matches total number of sequences
        assert len(self.lengths_arr) == self.total_number
        
    def __len__(self):
        return self.total_number
    
    def __getitem__(self, item):
        # Return sequence up to its actual length
        return self.ids_arr[item, :self.lengths_arr[item]]


class SequenceDataset(Dataset):
    """Dataset for handling tokenized sequences with padding and attention masks"""
    def __init__(self, ids_cache, max_seq_length):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length
        
    def __len__(self):  
        return len(self.ids_cache)

    def __getitem__(self, item):
        # Get token IDs for the sequence
        input_ids = self.ids_cache[item].tolist()
        
        # Truncate sequence to max length while preserving first and last tokens
        seq_length = min(self.max_seq_length-1, len(input_ids)-1)
        input_ids = [input_ids[0]] + input_ids[1:seq_length] + [input_ids[-1]]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1]*len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": item,
        }


class SubsetSeqDataset:
    """Dataset wrapper that only exposes a subset of another sequence dataset"""
    def __init__(self, subset: List[int], ids_cache, max_seq_length):
        self.subset = sorted(list(subset))  # Indices to include
        self.alldataset = SequenceDataset(ids_cache, max_seq_length)
        
    def __len__(self):  
        return len(self.subset)

    def __getitem__(self, item):
        return self.alldataset[self.subset[item]]


def load_rel(rel_path):
    """Load relevance data mapping query IDs to relevant passage IDs"""
    reldict = defaultdict(list)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        qid, _, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        reldict[qid].append((pid))
    return dict(reldict)
    

def load_rank(rank_path):
    """Load ranking data mapping query IDs to ranked passage IDs"""
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        qid, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)


def pack_tensor_2D(lstlst, default, dtype, length=None):
    """Convert list of lists into padded 2D tensor"""
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function(max_seq_length):
    """Returns a collate function that handles batching of sequences"""
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        # Use max_seq_length for first 10 batches to calibrate
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function  


class TrainInbatchDataset(Dataset):
    """Dataset for training with in-batch negatives"""
    def __init__(self, rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length):
        self.query_dataset = SequenceDataset(queryids_cache, max_query_length)
        self.doc_dataset = SequenceDataset(docids_cache, max_doc_length)
        self.reldict = load_rel(rel_file)  # Maps queries to relevant passages
        self.qids = sorted(list(self.reldict.keys()))

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        # Randomly select a relevant passage for the query
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        return query_data, passage_data


class TrainInbatchWithHardDataset(TrainInbatchDataset):
    """Dataset that includes hard negative examples during training"""
    def __init__(self, rel_file, rank_file, queryids_cache, 
            docids_cache, hard_num,
            max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length)
        self.rankdict = json.load(open(rank_file))  # Pre-computed rankings
        assert hard_num > 0
        self.hard_num = hard_num  # Number of hard negatives per query

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        # Sample hard negative passages from pre-computed rankings
        hardpids = random.sample(self.rankdict[str(qid)], self.hard_num)
        hard_passage_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
        return query_data, passage_data, hard_passage_data


class TrainInbatchWithRandDataset(TrainInbatchDataset):
    """Dataset that includes random negative examples during training"""
    def __init__(self, rel_file, queryids_cache, 
            docids_cache, rand_num,
            max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length)
        assert rand_num > 0
        self.rand_num = rand_num  # Number of random negatives per query

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        # Sample random passages as negatives
        randpids = random.sample(range(len(self.doc_dataset)), self.rand_num)
        rand_passage_data = [self.doc_dataset[randpid] for randpid in randpids]
        return query_data, passage_data, rand_passage_data


def single_get_collate_function(max_seq_length, padding=False):
    """Returns collate function for single sequences"""
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10 or padding:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function  


def dual_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    """Returns collate function for query-document pairs"""
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        # Create mask for irrelevant pairs (1 for irrelevant, 0 for relevant)
        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            }
        return input_data
    return collate_function  


def triple_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    """Returns collate function for query-document-negative triples"""
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        hard_doc_data, hard_doc_ids = doc_collate_func(sum([x[2] for x in batch], []))
        
        # Create masks for irrelevant pairs
        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        hard_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in hard_doc_ids ]
            for qid in query_ids]
            
        query_num = len(query_data['input_ids'])
        hard_num_per_query = len(batch[0][2])
        
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "other_doc_ids":hard_doc_data['input_ids'].reshape(query_num, hard_num_per_query, -1),
            "other_doc_attention_mask":hard_doc_data['attention_mask'].reshape(query_num, hard_num_per_query, -1),
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            "hard_pair_mask":torch.FloatTensor(hard_pair_mask),
            }
        return input_data
    return collate_function  
