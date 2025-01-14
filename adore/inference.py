import sys
import os
import torch
import faiss
import logging
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaConfig
from adore.model import RobertaDot
from adore.dataset import TextTokenIdsCache, SequenceDataset, get_collate_function
from adore.retrieve_utils import (
    construct_flatindex_from_embeddings, 
    index_retrieve, convert_index_to_gpu
)

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


def evaluate(args, model):
    """ Train the model """    
    dev_dataset = SequenceDataset(
            TextTokenIdsCache(args.preprocess_dir, f"{args.mode}-query"), 
            args.max_seq_length)
    collate_fn = get_collate_function(args.max_seq_length)
    batch_size = args.pergpu_eval_batch_size
    if args.n_gpu > 1:
        batch_size *= args.n_gpu
    dev_dataloader = DataLoader(dev_dataset, 
        batch_size= batch_size, collate_fn=collate_fn)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    qembedding_memmap = np.memmap(args.qmemmap_path, dtype="float32",
        shape=(len(dev_dataset), 768), mode="w+")
    with torch.no_grad():
        for step, (batch, qoffsets) in enumerate(tqdm(dev_dataloader)):
            batch = {k:v.to(args.model_device) for k, v in batch.items()}
            model.eval()            
            embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], 
                is_query=True)
            embeddings = embeddings.detach().cpu().numpy()
            qembedding_memmap[qoffsets] = embeddings
    return qembedding_memmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--dmemmap_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--pergpu_eval_batch_size", type=int, default=32)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--faiss_gpus", type=int, default=None, nargs="+")
    args = parser.parse_args()
    logger.info(args)

    assert os.path.exists(args.dmemmap_path)
    os.makedirs(args.output_dir, exist_ok=True)
    # Setup CUDA, GPU 
    args.use_gpu = torch.cuda.is_available() and not args.no_cuda
    args.model_device = torch.device(f"cuda:0" if args.use_gpu else "cpu")
    args.n_gpu = 1

    # Setup logging
    logger.warning("Model Device: %s, n_gpu: %s", args.model_device, args.n_gpu)
    config = RobertaConfig.from_pretrained(args.model_dir)
    model = RobertaDot.from_pretrained(args.model_dir, config=config)
        
    model.to(args.model_device)
    logger.info("Training/evaluation parameters %s", args)
    # Evaluation
    args.qmemmap_path = f"{args.output_dir}/{args.mode}.qembed.memmap"
    evaluate(args, model)
    
    doc_embeddings = np.memmap(args.dmemmap_path, 
        dtype=np.float32, mode="r").reshape(-1, model.output_embedding_size)

    query_embeddings = np.memmap(args.qmemmap_path, 
        dtype=np.float32, mode="r").reshape(-1, model.output_embedding_size)
    model = None
    torch.cuda.empty_cache()

    index = construct_flatindex_from_embeddings(doc_embeddings, None)
    if args.faiss_gpus:
        index = convert_index_to_gpu(index, args.faiss_gpus, False)
    else:
        faiss.omp_set_num_threads(32)
    nearest_neighbors = index_retrieve(index, query_embeddings, args.topk, batch=32)
    output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")
    with open(output_rank_file, 'w') as outputfile:
        for qid, neighbors in enumerate(nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx+1}\n")
      
#--The Methods below are the same as original, just adapted for easier use in other scripts--# 

       
def evaluate(
    preprocess_dir: str,
    mode: str,
    max_seq_length: int,
    pergpu_eval_batch_size: int,
    n_gpu: int,
    model_device: torch.device,
    qmemmap_path: str,
    model: torch.nn.Module
) -> np.ndarray:
    """The same evaluate method as in original adore.train.py but with arguments as parameters
    for easier use in other scripts.
    
    Args:
        preprocess_dir: Directory containing preprocessed data
        mode: Mode for evaluation (train/dev/test/lead)
        max_seq_length: Maximum sequence length
        pergpu_eval_batch_size: Batch size per GPU
        n_gpu: Number of GPUs to use
        model_device: Device to run model on
        qmemmap_path: Path to save query embeddings
        model: The model to evaluate
    
    Returns:
        np.ndarray: Memory-mapped array of query embeddings
    """
    dev_dataset = SequenceDataset(
            TextTokenIdsCache(preprocess_dir, f"{mode}-query"), 
            max_seq_length)
    collate_fn = get_collate_function(max_seq_length)
    batch_size = pergpu_eval_batch_size
    if n_gpu > 1:
        batch_size *= n_gpu
    dev_dataloader = DataLoader(dev_dataset, 
        batch_size=batch_size, collate_fn=collate_fn)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    qembedding_memmap = np.memmap(qmemmap_path, dtype="float32",
        shape=(len(dev_dataset), 768), mode="w+")
    with torch.no_grad():
        for step, (batch, qoffsets) in enumerate(tqdm(dev_dataloader)):
            batch = {k:v.to(model_device) for k, v in batch.items()}
            model.eval()            
            embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], 
                is_query=True)
            embeddings = embeddings.detach().cpu().numpy()
            qembedding_memmap[qoffsets] = embeddings
    return qembedding_memmap
         
                
def run_inference(
    model_dir: str,
    output_dir: str,
    preprocess_dir: str,
    mode: str,
    dmemmap_path: str,
    topk: int = 100,
    max_seq_length: int = 64,
    pergpu_eval_batch_size: int = 32,
    use_cuda: bool = True,
    faiss_gpus: list[int] = None
) -> str:
    """Run the complete inference pipeline and return path to results. This is the same
    as the original adore.inference.py but with arguments as parameters for easier use
    in other scripts.
    
    Args:
        model_dir: Directory containing the model
        output_dir: Directory to save outputs
        preprocess_dir: Directory containing preprocessed data
        mode: Mode for evaluation (train/dev/test/lead)
        dmemmap_path: Path to document embeddings memmap
        topk: Number of nearest neighbors to retrieve
        max_seq_length: Maximum sequence length
        pergpu_eval_batch_size: Batch size per GPU
        use_cuda: Whether to use CUDA if available
        faiss_gpus: List of GPU indices to use for FAISS
    
    Returns:
        str: Path to the output rankings file
    """
    
    if not os.path.exists(dmemmap_path):
        raise FileNotFoundError(f"Document embeddings not found at {dmemmap_path}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    use_gpu = torch.cuda.is_available() and use_cuda
    model_device = torch.device("cuda:0" if use_gpu else "cpu")
    n_gpu = 1 if use_gpu else 0

    # Initialize model
    config = RobertaConfig.from_pretrained(model_dir)
    model = RobertaDot.from_pretrained(model_dir, config=config)
    model.to(model_device)

    # Generate query embeddings
    qmemmap_path = f"{output_dir}/{mode}.qembed.memmap"
    evaluate(
        preprocess_dir=preprocess_dir,
        mode=mode,
        max_seq_length=max_seq_length,
        pergpu_eval_batch_size=pergpu_eval_batch_size,
        n_gpu=n_gpu,
        model_device=model_device,
        qmemmap_path=qmemmap_path,
        model=model
    )

    # Load embeddings
    doc_embeddings = np.memmap(dmemmap_path, 
        dtype=np.float32, mode="r").reshape(-1, model.output_embedding_size)
    query_embeddings = np.memmap(qmemmap_path, 
        dtype=np.float32, mode="r").reshape(-1, model.output_embedding_size)

    # Clear GPU memory
    model = None
    torch.cuda.empty_cache()

    # Build index and retrieve
    index = construct_flatindex_from_embeddings(doc_embeddings, None)
    if faiss_gpus:
        index = convert_index_to_gpu(index, faiss_gpus, False)
    else:
        faiss.omp_set_num_threads(32)
    
    nearest_neighbors = index_retrieve(index, query_embeddings, topk, batch=32)
    
    # Save results
    output_rank_file = os.path.join(output_dir, f"{mode}.rank.tsv")
    with open(output_rank_file, 'w') as outputfile:
        for qid, neighbors in enumerate(nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx+1}\n")
    
    return output_rank_file
                
