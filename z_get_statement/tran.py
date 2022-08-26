from tree import ASTNode
import os
import pickle
import json
import gzip
from pycparser import c_parser, c_generator
from collections import Counter
from tqdm import tqdm
from typing import List


def get_blocks(node, block_seq):
    children = node.children()
    name = node.__class__.__name__
    if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
        block_seq.append(ASTNode(node))
        if name is not 'For':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
    elif name is 'Compound':
        block_seq.append(ASTNode(name))
        for _, child in node.children():
            if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
        block_seq.append(ASTNode('End'))
    else:
        for _, child in node.children():
            get_blocks(child, block_seq)


def tree_to_index(node):
    result = [node.token]
    children = node.children
    for child in children:
        result.append(tree_to_index(child))
    return result


def get_frequency():
    print('load train dataset')
    dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone/'
    with gzip.open(os.path.join(dir, 'test.gzip'), 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    json_objects = json.loads(json_str)
    parser = c_parser.CParser()
    generator = c_generator.CGenerator()
    snippets = []
    function_id_checked = set()
    for obj in tqdm(json_objects):
        if obj["item_1"]["function_id"] in function_id_checked:
            continue
        else:
            function_id_checked.add(obj["item_1"]["function_id"])
        function = obj["item_1"]["function"]
        # print(function)
        code_lines = []
        try:
            ast = parser.parse(function)
            function_beautiful = generator.visit(ast)
            for line in function_beautiful.split('\n'):
                line = line.replace('{', '').replace('}', '').strip()
                if line == '':
                    continue
                code_lines.append(line)
        except:
            pass
        for i in range(len(code_lines) - 4):
            snippets.append(code_lines[i]+' '+code_lines[i + 1]+' '+ code_lines[i + 2] + ' '+ code_lines[i + 3] + ' '+ code_lines[i + 4])

    counter = Counter(snippets)
    # for key,value in counter.items():
    #     if value>2:
    #         print(key +"    "+str(value))
    return counter


def create_snippet_with_low_frequency():
    counter = get_frequency()
    dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone/'
    with gzip.open(os.path.join(dir, 'test.gzip'), 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    json_objects = json.loads(json_str)
    parser = c_parser.CParser()
    generator = c_generator.CGenerator()
    for obj in tqdm(json_objects):
        function = obj["item_1"]["function"]
        code_lines = []
        try:
            ast = parser.parse(function)
            function_beautiful = generator.visit(ast)
            for line in function_beautiful.split('\n'):
                line = line.replace('{', '').replace('}', '').strip()
                if line == '':
                    continue
                code_lines.append(line)
        except:
            pass
        for i in range(len(code_lines) - 4):
            new_function = code_lines[i]+' '+code_lines[i + 1]+' '+ code_lines[i + 2] + ' '+ code_lines[i + 3] +' '+ code_lines[i + 4]
            if counter[new_function] > 2:
                continue
            obj["item_1"]["function"] = new_function
            break

    json_bytes = json.dumps(json_objects).encode('utf-8')
    with gzip.open(filename=os.path.join(dir, 'test_snippet.gzip'), mode='w') as f:
        f.write(json_bytes)

if __name__ == '__main__':
    create_snippet_with_low_frequency()
