"""
evaluation.py
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sacrebleu import BLEU
from sklearn.metrics import f1_score
from typing import Dict, Any
from dataset_loader import DatasetLoader

class Evaluator:
    def __init__(self, model: torch.nn.Module, eval_loader: DataLoader, config: Dict[str, Any] = None):
        """
        Initialize the evaluator with the model, data loader, and configuration.
        
        Args:
            model: The Transformer model to evaluate.
            eval_loader: DataLoader for the evaluation dataset.
            config: Configuration settings, including beam search parameters and maximum output length.
        """
        self.model = model
        self.eval_loader = eval_loader
        self.config = config if config else {}
        
        # Set default evaluation parameters if not provided in config
        self.beam_size = self.config.get('beam_size', 4)
        self.alpha = self.config.get('alpha', 0.6)
        self.max_output_length = self.config.get('max_output_length', 50)
        
        self.device = next(model.parameters()).device
    
    def evaluate(self) -> Dict[str, float]:
        """
        Perform evaluation on the model using the evaluation dataset.
        
        Returns:
            Dict containing evaluation metrics (BLEU score for translation, F1 score for parsing).
        """
        metrics = {'BLEU': 0.0, 'F1': 0.0}
        
        # Evaluate translation task
        bleu = self._compute_bleu()
        metrics['BLEU'] = bleu
        
        # Evaluate constituency parsing task
        f1 = self._compute_f1()
        metrics['F1'] = f1
        
        return metrics
    
    def _compute_bleu(self) -> float:
        """
        Compute the BLEU score for the translation task using beam search.
        
        Returns:
            BLEU score as a float.
        """
        bleu = BLEU(beam_size=self.beam_size, max_length=self.max_output_length, length_penalty=self.alpha)
        
        references = []
        hypotheses = []
        
        for batch in self.eval_loader:
            src, tgt = batch['src'], batch['tgt']
            src = src.to(self.device)
            
            # Generate translations using beam search
            with torch.no_grad():
                outputs = self.model.translate(src, self.beam_size, self.max_output_length, self.alpha)
            
            # Convert outputs and targets to strings
            for output, target in zip(outputs, tgt):
                hypotheses.append(output)
                references.append(target)
        
        # Calculate BLEU score
        bleu_score = bleu.corpus_score(hypotheses, [references])
        return bleu_score
    
    def _compute_f1(self) -> float:
        """
        Compute the F1 score for the constituency parsing task.
        
        Returns:
            F1 score as a float.
        """
        all_preds = []
        all_labels = []
        
        for batch in self.eval_loader:
            inputs, labels = batch['input'], batch['label']
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.parse_constituency(inputs)
            
            # Convert outputs and labels to numpy arrays
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate F1 score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return f1
    
    @staticmethod
    def compute_bleu_score(hypotheses: List[str], references: List[str]) -> float:
        """
        Static method to compute BLEU score given hypotheses and references.
        
        Args:
            hypotheses: List of generated texts.
            references: List of reference texts.
            
        Returns:
            BLEU score as a float.
        """
        bleu = BLEU()
        return bleu.corpus_score(hypotheses, [references])
    
    @staticmethod
    def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Static method to compute F1 score given true and predicted labels.
        
        Args:
            y_true: True labels as a numpy array.
            y_pred: Predicted labels as a numpy array.
            
        Returns:
            F1 score as a float.
        """
        return f1_score(y_true, y_pred, average='weighted')
