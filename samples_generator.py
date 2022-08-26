import torch
import pickle
import random
import os
from utils.util import print_msg
from tqdm import tqdm
from typing import List

POSITIVE_LABEL_ID = 1
NEGATIVE_LABEL_ID = 0


class SnippetInferInstance:
    """Subcode Inference Data Instance.
    Represents a single sample in the subcode inference task.

    Attributes:
        snippet (str): Snippet (part of function) string.
        context (str): Context (function) string.
        label (int): Label.
        snippet_embd (torch.Tensor): Snippet embedding.
        context_embd (torch.Tensor): Context embedding.
    """

    def __init__(self, context_embd,snippet_embd,label:int):
        self.context_embd: torch.Tensor = context_embd
        self.snippet_embd: torch.Tensor = snippet_embd
        self.label: int = label



def generate_samples(
        func_embs,
        snippet_embs,
        n_samples: int,
) -> List[SnippetInferInstance]:
    samples:List[SnippetInferInstance] =[]
    for i in range(1250):
        assert func_embs[i]['id'] == snippet_embs[i]['id']
        samples.append(SnippetInferInstance(func_embs[i]['emb'],snippet_embs[i]['emb'],1))

    for i in range(1250):
        pos_id = func_embs[i]['id']
        index = random.randint(0, 1250 - 1)
        neg_id = func_embs[index]['id']
        while neg_id == pos_id:
            index = random.randint(0, 1250 - 1)
            neg_id = func_embs[index]['id']
        assert pos_id != neg_id
        samples.append(SnippetInferInstance(func_embs[i]['emb'],snippet_embs[index]['emb'],0))
    return samples

if __name__ == '__main__':
    random.seed(1337)
    dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone/'
    with open(file=os.path.join(dir,'ModelType.BiLSTM_snippet.pl'), mode='rb') as f:
        snippet_obj = pickle.load(f)

    names = ['ModelType.BiLSTM.pl', 'ModelType.GCN.pl', 'ModelType.GGNN.pl', 'ModelType.TreeLSTM.pl']
    for name in names:
        path = os.path.join(dir, name)
        with open(file=path, mode='rb') as f:
            func_obj = pickle.load(f)
        print_msg(f"{name} has {len(func_obj['embs'])}")

        samples = generate_samples(func_obj['embs'],snippet_obj['embs'], n_samples=2500)
        print(f"Generated {len(samples)} samples")
        save_path = os.path.join(dir,f"{name}_emb_and_BiLSTM_snippet.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(samples, f)
        print_msg(f"save {save_path}")
