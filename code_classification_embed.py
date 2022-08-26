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

def save(trainer, data_loader, gzip_path, save_path,proportion=1.0):
    embs = trainer.get_embedding_from_classification(data_loader)
    with gzip.open(gzip_path, 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    objs = json.loads(json_str)
    assert len(embs) == len(objs)
    new_objs = []
    if proportion<0.9:
        objs = objs[int(proportion*len(objs))]
    for index, obj in enumerate(objs):
        new_objs.append({
            'function_id': obj['function_id'],
            'function': obj['function'],
            "LSTM_emb": embs[index]
        })
    with open(save_path, 'wb') as f:
        pickle.dump(new_objs, f)
    print(f'save {len(new_objs)} samples')


def emb(config, running_mode):
    """
    Test the model using the config object
    :param config: Configuration Object
    :param vector_type: Type of Get Vectors, can be empty
    :return: Return the object
    """
    config.test_mode = True

    dataset = DatasetFactory().get_dataset(config, running_mode)
    trainer = TrainerFactory().get_trainer(config)
    print_msg("Using Trainer %s" % trainer.name)
    print_msg("Using Dataset %s" % dataset.name)

    # Start the Training
    trainer.setup_model()
    model_path = os.path.join(config.output_path, "model.pt")
    assert os.path.exists(model_path), "Model path %s does not exists" % model_path
    trainer.load_pymodel(model_path)
    print_msg("Getting Test Score")

    train_dl, valid_dl = dataset.get_dls()
    save(trainer, train_dl, 'data/poj104/train.gzip', 'data/poj104statement/lstm-train.pkl')
    save(trainer, valid_dl, 'data/poj104/val.gzip', 'data/poj104statement/lstm-val.pkl')
    # save(trainer,dataset.get_testing_dl(),'data/poj104/test.gzip','data/poj104statement/lstm.pkl')


def main(args):
    torch_setup(args.gpu_id)
    config = Config(args.config_path, test_mode=True)
    config.print_params()
    config.class_weight = False
    config.evaluate = args.evaluate
    vector_type = args.vector_type
    config.setup_vocab_dict()

    print_msg('getting embedding')
    emb(config, args.running_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config_path", help="Configuration Path, YML Path", default="")
    parser.add_argument("--running_mode", help="running_mode", default="train")
    parser.add_argument("--gpu_id", help="GPU ID", default="1")
    parser.add_argument("--vector_type", default="")
    parser.add_argument("--evaluate", help="Specify if you want look into each prediction result", action='store_true')
    args = parser.parse_args()
    main(args)
