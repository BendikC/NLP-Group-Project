import sys
import os
import time
import torch
import random
import faiss
import logging
import argparse
import subprocess
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup)

from adore.dataset import TextTokenIdsCache, SequenceDataset, load_rel, pack_tensor_2D


from corpus_management.load_corpus import load_encoded_corpus

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)    


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, output_dir, save_name, args, optimizer=None):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))


class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, 
            rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = load_rel(rel_file)

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_ids'] = self.reldict[item]
        return ret_val


def get_collate_function(max_seq_length):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
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
        qids = [x['id'] for x in batch]
        all_rel_pids = [x["rel_ids"] for x in batch]
        return data, all_rel_pids
    return collate_function  
    

gpu_resources = []

def load_index(passage_embeddings,  faiss_gpu_index):
    dim = passage_embeddings.shape[1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(passage_embeddings)
    if faiss_gpu_index:
        if len(faiss_gpu_index) == 1:
            res = faiss.StandardGpuResources()
            res.setTempMemory(128*1024*1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            global gpu_resources
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(128*1024*1024)
                gpu_resources.append(res)

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index


def train(args, model):
    """ Train the model """
    tb_writer = SummaryWriter(os.path.join(args.log_dir, 
        time.strftime("%b-%d_%H-%M-%S", time.localtime())))

    passage_embeddings, _ = load_encoded_corpus(for_train=True)
    
    args.train_batch_size = args.per_gpu_batch_size
    train_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "train-query"),
        os.path.join(args.preprocess_dir, "train-qrel.tsv"),
        args.max_seq_length
    )

    train_sampler = RandomSampler(train_dataset) 
    collate_fn = get_collate_function(args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    index = load_index(passage_embeddings, args.faiss_gpu_index)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_mrr, logging_mrr = 0.0, 0.0
    tr_recall, logging_recall = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)  

    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, all_rel_pids) in enumerate(epoch_iterator):

            batch = {k:v.to(args.model_device) for k, v in batch.items()}
            model.train()            
            model_output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], 
            )
            
            query_embeddings = model_output.pooler_output
            I_nearest_neighbor = index.search(
                    query_embeddings.detach().cpu().numpy(), args.neg_topk)[1]
            
            loss = 0
            for retrieve_pids, cur_rel_pids, qembedding in zip(
                I_nearest_neighbor, all_rel_pids, query_embeddings):
                target_labels = np.isin(retrieve_pids, cur_rel_pids).astype(np.int32)

                first_rel_pos = np.where(target_labels[:10])[0] 
                mrr = 1/(1+first_rel_pos[0]) if len(first_rel_pos) > 0 else 0

                tr_mrr += mrr/args.train_batch_size
                recall = 1 if mrr > 0 else 0
                tr_recall += recall / args.train_batch_size

                if np.sum(target_labels) == 0:
                    retrieve_pids = np.hstack([retrieve_pids, cur_rel_pids])
                    target_labels = np.hstack([target_labels, [True]*len(cur_rel_pids)])
                    assert len(retrieve_pids) == len(target_labels)

                target_labels = target_labels.reshape(-1, 1)
                rel_diff = target_labels - target_labels.T
                pos_pairs = (rel_diff > 0).astype(np.float32)
                num_pos_pairs = np.sum(pos_pairs, (0, 1))
                
                assert num_pos_pairs > 0
                neg_pairs = (rel_diff < 0).astype(np.float32)
                num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same

                pos_pairs = torch.FloatTensor(pos_pairs).to(args.model_device)
                neg_pairs = torch.FloatTensor(neg_pairs).to(args.model_device)
                
                topK_passage_embeddings = torch.FloatTensor(
                    passage_embeddings[retrieve_pids]).to(args.model_device)
                y_pred = (qembedding.unsqueeze(0) * topK_passage_embeddings).sum(-1, keepdim=True)
                sigma = 1

                C_pos = torch.log(1 + torch.exp(-sigma * (y_pred - y_pred.t())))
                C_neg = torch.log(1 + torch.exp(sigma * (y_pred - y_pred.t())))

                C = pos_pairs * C_pos + neg_pairs * C_neg
              
                arr = 1/(torch.arange(1, 1+len(y_pred)).float().to(y_pred.device))
                arr[args.metric_cut:] = 0
                weights = torch.abs(arr.view(-1,1) - arr.view(1, -1))
                C = C * weights
                cur_loss = torch.sum(C, (0, 1)) / num_pairs
                loss += cur_loss
            
            loss /= (args.train_batch_size * args.gradient_accumulation_steps)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('train/all_loss', cur_loss, global_step)
                    logging_loss = tr_loss

                    cur_mrr =  (tr_mrr - logging_mrr)/(
                        args.logging_steps * args.gradient_accumulation_steps)
                    tb_writer.add_scalar('train/mrr_10', cur_mrr, global_step)
                    logging_mrr = tr_mrr

                    cur_recall =  (tr_recall - logging_recall)/(
                        args.logging_steps * args.gradient_accumulation_steps)
                    tb_writer.add_scalar('train/recall_10', cur_recall, global_step)
                    logging_recall = tr_recall

                if args.save_steps > 0 and global_step % args.save_steps == 0:                    
                    save_model(model, args.model_save_dir, 'ckpt-{}'.format(global_step), args)
        
        save_model(model, args.model_save_dir, 'epoch-{}'.format(epoch_idx+1), args)


def run_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_cut", type=int, default=None)
    parser.add_argument("--pembed_path", type=str, required=True)
    parser.add_argument("--model_save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--neg_topk", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--per_gpu_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", default=2000, type=int)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--save_steps", type=int, default=5000000) # not use
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=6, type=int)

    parser.add_argument("--model_gpu_index", type=int, default=0)
    parser.add_argument("--faiss_gpu_index", type=int, default=[], nargs="+")
    parser.add_argument("--faiss_omp_num_threads", type=int, default=32)
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.faiss_omp_num_threads)

    return args


# def main():
#     args = run_parse_args()
#     # Setup CUDA, GPU 
#     args.model_device = torch.device(f"cuda:{args.model_gpu_index}")
#     args.n_gpu = torch.cuda.device_count()

#     # Setup logging
#     logger.warning("Model Device: %s, n_gpu: %s", args.model_device, args.n_gpu)

#     # Set seed
#     set_seed(args)

#     # Initialize Contriever model
#     model = Contriever(DenseHyperParams)

#     model.to(args.model_device)
#     logger.info("Training/evaluation parameters %s", args)
    
#     os.makedirs(args.model_save_dir, exist_ok=True)
#     train(args, model)
    
# if __name__ == "__main__":
#     main()


#--Functions below are the same as in original, just adapted for easier use in other scripts--#
 
def run_training(
    model,
    pembed_path: str,
    model_save_dir: str,
    log_dir: str,
    preprocess_dir: str,
    model_gpu_index: int = 0,
    faiss_gpu_index: list = None,
    faiss_omp_num_threads: int = 8,
    metric_cut: int = None,
    neg_topk: int = 100,
    max_seq_length: int = 64,
    per_gpu_batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    warmup_steps: int = 1000,
    seed: int = 42,
    save_steps: int = 5000000,
    logging_steps: int = 100,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    num_train_epochs: int = 6
):
    """Same functionality as main() but takes arguments as parameters instead of parsing them,
    for easier use in other scripts.

    Args:
        model: Contriever model used as the base model for training
        pembed_path: Path to passage embeddings
        model_save_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
        preprocess_dir: Directory containing preprocessed data
        model_gpu_index: GPU index to use for model (default: 0)
        faiss_gpu_index: List of GPU indices to use for FAISS (default: None)
        faiss_omp_num_threads: Number of OpenMP threads for FAISS (default: 32)
        metric_cut: Cut-off for evaluation metrics (default: None)
        neg_topk: Number of negative samples to use (default: 200)
        max_seq_length: Maximum sequence length (default: 64)
        per_gpu_batch_size: Batch size per GPU (default: 32)
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 1)
        warmup_steps: Number of warmup steps (default: 2000)
        seed: Random seed (default: 42)
        save_steps: Save checkpoint every N steps (default: 5000000)
        logging_steps: Log metrics every N steps (default: 100)
        learning_rate: Learning rate (default: 5e-6)
        weight_decay: Weight decay (default: 0.01)
        adam_epsilon: Epsilon for Adam optimizer (default: 1e-8)
        max_grad_norm: Maximum gradient norm (default: 1.0)
        num_train_epochs: Number of training epochs (default: 6)
    """
    
    # Setup CUDA, GPU
    model_device = torch.device(f"cuda:{model_gpu_index}")
    n_gpu = torch.cuda.device_count()
    
    # Add check for log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    logger.warning("Model Device: %s, n_gpu: %s", model_device, n_gpu)

    # Create args object to maintain compatibility with existing code
    args = argparse.Namespace(
        pembed_path=pembed_path,
        model_save_dir=model_save_dir,
        log_dir=log_dir,
        preprocess_dir=preprocess_dir,
        model_gpu_index=model_gpu_index,
        faiss_gpu_index=faiss_gpu_index if faiss_gpu_index else [],
        faiss_omp_num_threads=faiss_omp_num_threads,
        metric_cut=metric_cut,
        neg_topk=neg_topk,
        max_seq_length=max_seq_length,
        per_gpu_batch_size=per_gpu_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        seed=seed,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        model_device=model_device,
        n_gpu=n_gpu
    )

    # Set FAISS threads
    faiss.omp_set_num_threads(faiss_omp_num_threads)

    # Set seed
    set_seed(args)

    model.to(model_device)
    logger.info("Training/evaluation parameters %s", args)
    
    os.makedirs(model_save_dir, exist_ok=True)
    train(args, model)
