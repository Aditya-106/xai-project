"""
Entailment scoring for SEA-CoT.

Uses a proper NLI model (DeBERTa-v3-base fine-tuned on MNLI/FEVER/ANLI)
to compute entailment probability between an explanation (premise)
and the question+answer (hypothesis).

The paper uses DeBERTa-large-MNLI. We default to cross-encoder/nli-deberta-v3-base
which is smaller but still produces well-calibrated NLI scores,
and runs efficiently on CPU/Apple MPS.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EntailmentScorer:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        """
        Initialize the entailment scorer.

        Args:
            model_name: HuggingFace model name. Recommended options:
                - "cross-encoder/nli-deberta-v3-base" (default, ~184M params, CPU-friendly)
                - "microsoft/deberta-large-mnli" (paper's model, ~350M params)
                - "cross-encoder/nli-deberta-v3-large" (larger, more accurate)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cpu")  # NLI models run fast on CPU
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

        # Determine label mapping based on model
        # cross-encoder/nli-deberta-v3-base: 0=contradiction, 1=entailment, 2=neutral
        # microsoft/deberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment
        if "cross-encoder" in model_name:
            self.entailment_idx = 1
        else:
            # DeBERTa-large-MNLI and similar: entailment is label 2
            self.entailment_idx = 2

    def calculate_entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        Calculates P(entailment | premise, hypothesis).

        Args:
            premise: The explanation text (ê_i)
            hypothesis: The question + predicted answer (x ⊕ ŷ)

        Returns:
            Entailment probability in [0, 1]
        """
        if not premise or not hypothesis:
            return 0.0

        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt",
            truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        entailment_prob = probs[0][self.entailment_idx].item()
        return entailment_prob

# Global singleton for easy import
_scorer = None

def calculate_entailment_score(premise: str, hypothesis: str) -> float:
    """Convenience function using a global singleton scorer."""
    global _scorer
    if _scorer is None:
        _scorer = EntailmentScorer()
    return _scorer.calculate_entailment_score(premise, hypothesis)
