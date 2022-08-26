from dataset.poj104clone_dataset import POJ104Clone
from dataset.poj_dataset import POJ104
from dataset.devign_dataset import Devign


class DatasetFactory:
    name = "DatasetFactory"

    @staticmethod
    def get_dataset(config,running_mode='train',is_snippet='False'):
        """
        Factory Function to retrieve the necessary dataloader object
        A specific dataloader for a dataset
        :param config: Configuration object
        :return:
        """
        logger = config.logger
        dataset_name = config.dataset['name']
        logger.info("Dataset: %s" % dataset_name)
        if dataset_name == "online-judge":
            dataloader = POJ104(config, running_mode=running_mode)
        elif dataset_name == "devign":
            dataloader = Devign(config)
        elif dataset_name == "poj104clone":
            dataloader = POJ104Clone(config, running_mode=running_mode, is_snippet=is_snippet)
        else:
            raise SystemExit(Exception("Dataset Name %s is not found." % dataset_name))
        return dataloader
