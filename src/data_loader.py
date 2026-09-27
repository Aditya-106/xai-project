"""
Dataset loader for StrategyQA, OpenBookQA, and QASC.
"""
import json
import os
import argparse
from typing import List, Dict, Any
from datasets import load_dataset


def load_strategyqa(download: bool = False) -> List[Dict[str, Any]]:
    """Loads StrategyQA dataset."""
    dataset = load_dataset('wics/strategy-qa', split='train', trust_remote_code=True)
    processed = []
    for item in dataset:
        processed.append({
            'id': str(item.get('id', hash(item['question']))),
            'question': item['question'],
            'choices': ['yes', 'no'],
            'answer': 'yes' if item['answer'] else 'no',
            'answer_idx': 0 if item['answer'] else 1
        })
    return processed

def load_openbookqa(download: bool = False) -> List[Dict[str, Any]]:
    """Loads OpenBookQA dataset."""
    dataset = load_dataset('allenai/openbookqa', 'main', split='train', trust_remote_code=True)
    processed = []
    for item in dataset:
        choices = item['choices']['text']
        labels = item['choices']['label']
        answer_label = item['answerKey']
        answer_idx = labels.index(answer_label)
        
        processed.append({
            'id': str(item['id']),
            'question': item['question_stem'],
            'choices': choices,
            'answer': choices[answer_idx],
            'answer_idx': answer_idx
        })
    return processed

def load_qasc(download: bool = False) -> List[Dict[str, Any]]:
    """Loads QASC dataset."""
    dataset = load_dataset('allenai/qasc', split='train', trust_remote_code=True)
    processed = []
    for item in dataset:
        choices = item['choices']['text']
        labels = item['choices']['label']
        answer_label = item['answerKey']
        answer_idx = labels.index(answer_label)
        
        processed.append({
            'id': str(item['id']),
            'question': item['formatted_question'],
            'choices': choices,
            'answer': choices[answer_idx],
            'answer_idx': answer_idx
        })
    return processed

def get_dataset(name: str, split: str = 'test') -> List[Dict[str, Any]]:
    """Load a dataset by name. Convenience function for experiment scripts."""
    loaders = {
        'strategyqa': load_strategyqa,
        'obqa': load_openbookqa,
        'qasc': load_qasc
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(loaders.keys())}")
    return loaders[name]()


def save_dataset(dataset: List[Dict[str, Any]], name: str, base_dir: str = 'data') -> None:
    """Saves processed dataset to JSONL."""
    os.makedirs(os.path.join(base_dir, name), exist_ok=True)
    out_path = os.path.join(base_dir, name, 'processed.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {name} to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help="Download datasets")
    args = parser.parse_args()

    if args.download:
        sqa = load_strategyqa(download=True)
        save_dataset(sqa, 'strategyqa')

        obqa = load_openbookqa(download=True)
        save_dataset(obqa, 'openbookqa')

        qasc = load_qasc(download=True)
        save_dataset(qasc, 'qasc')

if __name__ == '__main__':
    main()
