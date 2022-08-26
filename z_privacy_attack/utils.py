from typing import Dict


def format_metric_string(metric_dict: Dict[str, float]) -> str:
    metric_str = ''
    for k, v in metric_dict.items():
        metric_str += f'| {k} {v:.4f} '
    metric_str += '|'
    return metric_str


def merge_metric_dict(*metric_dict_list: Dict[str, float]) -> Dict[str, float]:
    metric_dict = {}
    for metric_dict_ in metric_dict_list:
        for k, v in metric_dict_.items():
            if k not in metric_dict:
                metric_dict[k] = v
            else:
                metric_dict[k] += v
    return metric_dict
