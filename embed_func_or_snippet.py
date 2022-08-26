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


def get_func_ids():
    path = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone/test_snippet.gzip'
    with gzip.open(path, 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    json_objects = json.loads(json_str)
    ids = []
    for obj in json_objects:
        ids.append(obj["item_1"]["function_id"])
    assert len(ids) == 5000
    return ids


def test(config, running_mode, is_snippet):
    """
    Test the model using the config object
    :param config: Configuration Object
    :param vector_type: Type of Get Vectors, can be empty
    :return: Return the object
    """
    config.test_mode = True
    # Retrieve and Format the dataset
    # TF-IDF does not have testing method
    if config.model_type in [ModelType.XGBoost, ModelType.SVM, ModelType.NaiveBayes]:
        return

    dataset = DatasetFactory().get_dataset(config, running_mode, is_snippet)
    trainer = TrainerFactory().get_trainer(config)
    print_msg("Using Trainer %s" % trainer.name)
    print_msg("Using Dataset %s" % dataset.name)

    # Start the Training
    trainer.setup_model()
    model_path = os.path.join(config.output_path, "model.pt")
    assert os.path.exists(model_path), "Model path %s does not exists" % model_path
    trainer.load_pymodel(model_path)
    print_msg("Getting Test Score")
    return trainer.get_embedding(dataset)


def main(args):
    torch_setup(args.gpu_id)
    config = Config(args.config_path, test_mode=True)
    config.print_params()
    config.class_weight = False
    config.evaluate = args.evaluate
    vector_type = args.vector_type
    config.setup_vocab_dict()
    print_msg('getting embedding')
    embs = test(config, args.running_mode, args.is_snippet)
    ids = get_func_ids()
    print_msg(f"embs length: {len(embs)} and ids length: {len(ids)}")
    func_id_emb = []
    duplicated_ids = set()
    for index in range(len(embs)):
        if ids[index] in duplicated_ids:
            continue
        duplicated_ids.add(ids[index])
        func = {}
        func["id"] = ids[index]
        func["emb"] = embs[index]
        func_id_emb.append(func)

    print_msg(f"get {len(func_id_emb)} embeddings")
    data_obj = {
        "name": f"poj104clone_test_full_emb_{config.model_type}",
        "embs": func_id_emb
    }
    data_dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone'
    if args.is_snippet:
        save_path = os.path.join(data_dir, f'{config.model_type}_snippet.pl')
    else:
        save_path = os.path.join(data_dir, f'{config.model_type}.pl')
    print(f'save to {save_path}')
    with open(file=save_path, mode='wb') as f:
        pickle.dump(data_obj, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config_path", help="Configuration Path, YML Path", default="")
    parser.add_argument("--running_mode", help="running_mode", default="test")
    parser.add_argument("--is_snippet", help="is_snippet", default="False")
    parser.add_argument("--gpu_id", help="GPU ID", default="1")
    parser.add_argument("--vector_type", default="")
    parser.add_argument("--evaluate", help="Specify if you want look into each prediction result", action='store_true')
    args = parser.parse_args()
    main(args)
