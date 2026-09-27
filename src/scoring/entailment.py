"""
Entailment scoring for SEA-CoT.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EntailmentScorer:
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def calculate_entailment_score(self, premise: str, hypothesis: str) -> float:
        """Calculates entailment probability."""
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
        # For deberta-large-mnli, label 2 is entailment (check HF model specific labels for accuracy)
        # 0: contradiction, 1: neutral, 2: entailment
        entailment_prob = probs[0][-1].item()
        return entailment_prob

# Global singleton for easy import
_scorer = None

def calculate_entailment_score(premise: str, hypothesis: str) -> float:
    global _scorer
    if _scorer is None:
        _scorer = EntailmentScorer()
    return _scorer.calculate_entailment_score(premise, hypothesis)
