import os
import logging
import yaml
from typing import Dict, Any
from .model import Model
from .trainer import Trainer
from .evaluation import Evaluator

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(
        data_path=config['dataset']['data_path'],
        batch_size=config['training']['batch_size']
    )
    
    # Create data loaders
    data_loaders = dataset_loader.load_data()
    
    # Initialize model
    model = Model(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    # Initialize evaluator
    evaluator = Evaluator(
        model=model,
        eval_loader=data_loaders['test']
    )
    
    # Initialize trainer
    trainer = Trainer(
        config=config
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = evaluator.evaluate()
    
    # Print evaluation results
    logger.info(f"Final evaluation metrics: {metrics}")

if __name__ == '__main__':
    main()
