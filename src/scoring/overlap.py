"""
Token overlap scoring for SEA-CoT.
"""
import nltk
from nltk.corpus import stopwords
from typing import Set
import re

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def tokenize_and_remove_stopwords(text: str) -> Set[str]:
    stop_words = set(stopwords.words('english'))
    # Basic tokenization
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set([t for t in tokens if t not in stop_words])

def calculate_overlap_score(explanation: str, question_answer: str) -> float:
    """
    Calculate IoU between explanation tokens and question+answer tokens.
    S_o = |ê_i ∩ (x ⊕ ŷ)| / |ê_i ∪ (x ⊕ ŷ)|
    """
    set_e = tokenize_and_remove_stopwords(explanation)
    set_qa = tokenize_and_remove_stopwords(question_answer)
    
    intersection = set_e.intersection(set_qa)
    union = set_e.union(set_qa)
    
    if len(union) == 0:
        return 0.0
        
    return len(intersection) / len(union)
