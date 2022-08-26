# Learning Program
Semantics with Code Representations: An Empirical Study

This repository contains the code and data in our paper, "Learning Program
Semantics with Code Representations: An Empirical Study"
published in SANER'2022. It includes POJ104Clone and POJ dataset. 

* Clone Detection - Pairwise Clone Detection
* Code Classification - Classify Code in their respective label
* Vulnerability Detection - See Devign

## Dataset

I had uploaded the dataset to google drive. You can download it [here](https://drive.google.com/file/d/1U4xQnrbym8T8QjGzTRIqqPdx7VwPP3H2/view?usp=sharing)

## Train

You can train the model with the sample command:
```shell script
python3 -u /home/jingkai/projects/cit/train.py --config_path ./ymls/clone_detection/tfidf/naivebayes.yml
```
Please look into `./ymls/<tasks>/*.yml` for setting the configurations.

## Citation
If you find this repository useful in your research, please consider citing it:
```
@inproceedings{siow2022learning,
  title={Learning Program Semantics with Code Representations: An Empirical Study},
  author={Jing Kai, Siow and Shangqing, Liu and Xiaofei, Xie and Guozhu, Meng and Yang, Liu},
  booktitle={Proceedings of the 29th IEEE International Conference onSoftware Analysis, Evolution and Reengineering},
  year={2022}
}
```


# dataset
test dataset : json list, len == 5000
```json
[
  {
    'item_1': {'function_id': '1','jsgraph': {'graph': [[1,2,0],...],'function': ''}},
    'item_2': {},
    'target': 1/0
    
  }
]
```

## POJ104

```json
[
  {
    "function_id": "str",
    "target": "int",
    "jsgraph": {
      "graph": [
        [
          1,
          2,
          0
        ]
      ],
      "node_features":{ "0": ["Function","","0","False"]
      }
    },
    "jsgraph_file_path":"str",
    "function": "str",
    "graph_size":"int",
    "cfile_path":"str"
  }
]
```

# load dataset steps（以Tree-LSTM为例）
* DatasetFactory().get_dataset(config)
* dataloader = POJ104
* BaseDataset
  * 定义 self.dataformatter = FormatterFactory().get_formatter(self.config)
    * TreeLSTMFormatter
    * 在BaseFormatter中，定义collate_fn:collate_graph_for_classification
  * 从gzip导入json数据 train/val/test load from json
  * self.format_data()
  * self._format(train/val/test)
  * datapoints.append(self.dataformatter.format(item, self.get_vocabs()))

# trained_model
## code_classification
### treelstm
20220727-163602: textual