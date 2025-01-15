"""
Handles adore model training, ngl I have no idea what this is going to look like so
i'll leave it up to whoever is implementing it. Hopefully it's just plug and play, 
and maybe this file isn't even necessary.
"""

import os
import torch
import logging
from adore.train import run_training
from dexter.retriever.dense.Contriever import Contriever
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams

logger = logging.getLogger(__name__)

class AdoreTrainer:
    """
    Handles training of the ADORE dense retrieval model.
    """
    
    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize the ADORE trainer.

        Args:
            config_path (str): Path to config file
        """
        self.config_path = config_path

    def train_model(self):
        """
        Initiaizes required parameters for training and runs the training.
        """
        
        logger.info("Initializing parameters for training")
        model = Contriever(DenseHyperParams(
                query_encoder_path=self.config["Query-Encoder"].get("query_encoder_path"),
                document_encoder_path=self.config["Document-Encoder"].get("document_encoder_path")
            )).context_encoder.cuda()
        pembed_path = "data/embeddings/passage_embeddings"
        model_save_dir = "model_checkpoints/adore"
        log_dir = "logs/adore"
        preprocess_dir = "data/preprocessed"
        model_gpu_index = 0
        faiss_gpu_index = [0]
        faiss_omp_num_threads = 8
        per_gpu_batch_size = 16
        learning_rate = 5e-6
        
        logger.info("Starting ADORE model training")
    
        # Run the training
        run_training(
            model=model,
            pembed_path=pembed_path,
            model_save_dir=model_save_dir,
            log_dir=log_dir,
            preprocess_dir=preprocess_dir,
            model_gpu_index=model_gpu_index,
            # faiss_gpu_index=faiss_gpu_index,
            # faiss_omp_num_threads=faiss_omp_num_threads,
            per_gpu_batch_size=per_gpu_batch_size,
            learning_rate=learning_rate
        )

        logger.info("ADORE model training completed")
