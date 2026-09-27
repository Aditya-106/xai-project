"""
Dataset loader for StrategyQA, OpenBookQA, and QASC.
Compatible with latest HuggingFace datasets library.
"""
import json
import os
import argparse
from typing import List, Dict, Any


def load_strategyqa(download: bool = False) -> List[Dict[str, Any]]:
    """Loads StrategyQA dataset."""
    from datasets import load_dataset
    # Use the ChilleD version which has proper Parquet format
    try:
        dataset = load_dataset('ChilleD/StrategyQA', split='test')
    except Exception:
        try:
            dataset = load_dataset('tasksource/strategyqa', split='test')
        except Exception:
            # Fallback: load from train split of another source
            dataset = load_dataset('metaeval/strategy-qa', split='train')

    processed = []
    for i, item in enumerate(dataset):
        # Handle different field names across dataset versions
        question = item.get('question', item.get('input', ''))
        answer_raw = item.get('answer', item.get('target', item.get('label', '')))

        # Normalize answer to 'yes'/'no'
        if isinstance(answer_raw, bool):
            answer = 'yes' if answer_raw else 'no'
        elif isinstance(answer_raw, str):
            answer = answer_raw.strip().lower()
            if answer not in ('yes', 'no'):
                answer = 'yes' if answer in ('true', '1') else 'no'
        elif isinstance(answer_raw, int):
            answer = 'yes' if answer_raw == 1 else 'no'
        else:
            answer = 'yes'

        processed.append({
            'id': str(item.get('id', i)),
            'question': question,
            'choices': ['yes', 'no'],
            'answer': answer,
            'answer_idx': 0 if answer == 'yes' else 1
        })
    return processed


def load_openbookqa(download: bool = False) -> List[Dict[str, Any]]:
    """Loads OpenBookQA dataset."""
    from datasets import load_dataset
    dataset = load_dataset('allenai/openbookqa', 'main', split='test')
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
    from datasets import load_dataset
    dataset = load_dataset('allenai/qasc', split='validation')
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
    """Load a dataset by name."""
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
    print(f"Saved {len(dataset)} examples to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help="Download datasets")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'strategyqa', 'obqa', 'qasc'])
    args = parser.parse_args()

    if args.download:
        if args.dataset in ('all', 'strategyqa'):
            print("Loading StrategyQA...")
            sqa = load_strategyqa(download=True)
            save_dataset(sqa, 'strategyqa')

        if args.dataset in ('all', 'obqa'):
            print("Loading OpenBookQA...")
            obqa = load_openbookqa(download=True)
            save_dataset(obqa, 'openbookqa')

        if args.dataset in ('all', 'qasc'):
            print("Loading QASC...")
            qasc = load_qasc(download=True)
            save_dataset(qasc, 'qasc')


if __name__ == '__main__':
    main()
