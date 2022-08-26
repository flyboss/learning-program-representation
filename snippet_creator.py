import os
import json
import gzip
import argparse
from configs.config import Config
from configs.model_type import ModelType
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.util import print_msg, torch_setup, load_vocab_dict, get_pretty_metric
import pickle
from pycparser import c_parser, c_generator
from collections import Counter

def create_snippet():
    dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone/'
    with gzip.open(os.path.join(dir,'test.gzip'), 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    json_objects = json.loads(json_str)
    for obj in json_objects:
        function = '\n'.join(obj["item_1"]["function"].split("\n")[:3])
        print(function)
        obj["item_1"]["function"] = function

    json_bytes = json.dumps(json_objects).encode('utf-8')
    with gzip.open(filename=os.path.join(dir,'test_snippet.gzip'),mode='w') as f:
        f.write(json_bytes)



def get_emb_length():
    dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone/'
    names = ['ModelType.BiLSTM.pl','ModelType.GCN.pl','ModelType.GGNN.pl','ModelType.TreeLSTM.pl']
    for name in names:
       path = os.path.join(dir,name)
       with open(file = path,mode='rb') as f:
           obj = pickle.load(f)
       print_msg(f"{name} has {len(obj['embs'])}")

def read_classi():
    path = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104/test.gzip'
    with gzip.open(path, 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    objs = json.loads(json_str)
    print('finish')

if __name__ == '__main__':
    read_classi()
    # create_snippet()