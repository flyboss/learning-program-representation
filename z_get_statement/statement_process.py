from pycparser import c_parser, c_generator, c_ast
from tree import ASTNode
from typing import List, Set, FrozenSet
import gzip
import os
import json
from tqdm import tqdm
from collections import Counter
import pickle
import pandas as pd

poj104statement_dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104statement/'

test_code = r"""
int main(int x)
{
struct pp o
char a[100],t;
char b[81][100];
char *p[10];
char *q;
char **qq;


scanf("%s%s",a,b);
int m;
m=strlen(a);

char *qwer[m];
return 0;
}
    """


class DeclVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.var_name = dict()

    def visit_Decl(self, node):
        if isinstance(node.type, c_ast.FuncDecl):
            pass
        elif isinstance(node.type, c_ast.ArrayDecl):
            array_decl = node.type
            if isinstance(array_decl.type, c_ast.ArrayDecl):
                self.visit_Decl(array_decl)
            elif isinstance(array_decl.type, c_ast.PtrDecl):
                old_name = array_decl.type.type.declname
                new_name = array_decl.type.type.type.names[0] + '_ptr_array'
                self.var_name[old_name] = new_name
                array_decl.type.type.declname = new_name
            else:
                old_name = array_decl.type.declname
                new_name = array_decl.type.type.names[0] + '_array'
                self.var_name[old_name] = new_name
                array_decl.type.declname = new_name
            if isinstance(array_decl.dim, c_ast.Constant):
                array_decl.dim.value = 'array_dim'
            elif isinstance(array_decl.dim, c_ast.ID):
                array_decl.dim.name = self.var_name.get(array_decl.dim.name, 'default')
        elif isinstance(node.type, c_ast.PtrDecl):
            ptr_decl = node.type
            while not isinstance(ptr_decl.type, c_ast.TypeDecl):
                ptr_decl = ptr_decl.type
            type_decl = ptr_decl.type
            old_name = type_decl.declname
            new_name = type_decl.type.names[0] + '_ptr'
            self.var_name[old_name] = new_name
            type_decl.declname = new_name
        elif isinstance(node.type, c_ast.TypeDecl):
            type_decl = node.type
            if isinstance(type_decl.type, c_ast.Struct):
                struct_type = type_decl.type
                old_name = struct_type.name
                new_name = 'struct'
                self.var_name[old_name] = new_name
                struct_type.name = new_name
            else:
                old_name = type_decl.declname
                new_name = type_decl.type.names[0]
                self.var_name[old_name] = new_name
                type_decl.declname = new_name
                self.visit_Constant(node)

    def visit_ID(self, node):
        node.name = self.var_name.get(node.name, 'default')

    def visit_Constant(self, node):
        if node.type == 'string':
            node.value = 'constant_str'
        elif node.type == 'int':
            node.value = 'constant_int'
        elif node.type == 'double':
            node.value = 'constant_double'
        elif node.type == 'char':
            node.value = 'constant_ch'


class ProcessFunc:
    def __init__(self, debug=False):
        self.parser = c_parser.CParser()
        self.generator = c_generator.CGenerator()
        self.debug = debug
        self.parse_fail = []
        self.visit_fail = []

    def rewrite_func(self, func: str, index: int):
        try:
            ast = self.parser.parse(func)
        except Exception:
            # print('parse fail')
            self.parse_fail.append(index)
            return func
            # exit(-1)

        # remove_func = []
        # for func in ast.ext:
        #     if func.decl.name != 'main':
        #         remove_func.append(func)
        #
        # for func in remove_func:
        #     ast.ext.remove(func)

        if self.debug:
            ast.show()
        try:
            visitor = DeclVisitor()
            visitor.visit(ast)
            return self.generator.visit(ast)
        except:
            # print('visit fail')
            self.visit_fail.append(index)
            return func

    def get_statements(self, func: str, index: int) -> List[str]:
        func = self.rewrite_func(func, index)
        statements = []
        for line in func.split('\n'):
            for ch in ['{', '}']:
                line = line.replace(ch, '')
            line = line.strip()
            if line == '':
                continue
            statements.append(line)
        return statements

    def get_statements_sets(self, func: str, index: int) -> Set[FrozenSet[str]]:
        func = self.rewrite_func(func, index)
        statements = set()
        for line in func.split('\n'):
            for ch in ['{', '}']:
                line = line.replace(ch, '')
            line = line.strip()
            if line == '':
                continue
            statements.add(frozenset(line.split(' ')))
        return statements


def create_statement_table():
    process_func = ProcessFunc()
    with open(os.path.join(poj104statement_dir, 'lstm-train.pkl'), 'rb') as f:
        objs = pickle.load(f)

    # statements = []
    statements_set = set()
    attack_training_sample_number = int(len(objs) * 0.5)
    print(f'lstm-train has {len(objs)} samples, use {attack_training_sample_number} to attack training')

    for index, obj in enumerate(tqdm(objs[:attack_training_sample_number])):
        function = obj["function"]
        # statements.extend(process_func.get_statements(function, index))
        statements_set.update(process_func.get_statements_sets(function, index))

    print(f'parse fail {len(process_func.parse_fail)}')
    print(process_func.parse_fail)
    print(f'visit fail {len(process_func.visit_fail)}')
    print(process_func.visit_fail)

    print(len(statements_set))
    # counter = Counter(statements)
    # print(len(counter.items()))
    # with open(file=os.path.join(poj104statement_dir, 'statement_table.pkl'), mode='wb') as f:
    #     pickle.dump(counter, f)
    # print("save statement_table.pkl")


def comment_remover(code: str):
    import re
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, code)


def review_statement_table():
    with open(file=os.path.join(poj104statement_dir, 'statement_table.pkl'), mode='rb') as f:
        counter = pickle.load(f)
    data = {'name': [], 'count': []}
    for k, v in counter.most_common(len(counter.items())):
        data['name'].append(k)
        data['count'].append(v)
    df = pd.DataFrame(data)
    list_bin = [i for i in range(0,110,10)]
    cut_data = pd.cut(df['count'],list_bin)
    cut_data = cut_data.value_counts().sort_index()
    print(cut_data)
    sum = 0
    for i in cut_data.values:
        sum+=i
    print(f"(100,)\t{len(counter.items())-sum}")


def review_function_statement():
    with open(file=os.path.join(poj104statement_dir, 'lstm-train.pkl'), mode='rb') as f:
        objs = pickle.load(f)
    data = {'count': []}
    for index,obj in tqdm(enumerate(objs)):
        if 'label' in obj:
            data['count'].append(obj['label'].count(1))
        else:
            break
    df = pd.DataFrame(data)
    list_bin = [i for i in range(0, 100, 2)]
    cut_data = pd.cut(df['count'], list_bin)
    cut_data = cut_data.value_counts().sort_index()
    print(cut_data)


def create_train_test_dataset():
    with open(file=os.path.join(poj104statement_dir, 'statement_table.pkl'), mode='rb') as f:
        counter: Counter = pickle.load(f)

    for name in ['lstm-train.pkl', 'lstm-val.pkl']:
        process_func = ProcessFunc()
        print(name)
        with open(file=os.path.join(poj104statement_dir, name), mode='rb') as f:
            objs = pickle.load(f)
        if name == 'lstm-train.pkl':
            sample_number = int(len(objs) * 0.5)
        else:
            sample_number = len(objs)
        for index, obj in enumerate(tqdm(objs[:sample_number])):
            statements = process_func.get_statements(obj['function'], index)
            label = []
            for k, _ in counter.most_common():
                if k in statements:
                    label.append(1)
                else:
                    label.append(0)
            obj['label'] = label
        print(f'parse fail {len(process_func.parse_fail)}')
        print(process_func.parse_fail)
        print(f'visit fail {len(process_func.visit_fail)}')
        print(process_func.visit_fail)
        with open(file=os.path.join(poj104statement_dir, name), mode='wb') as f:
            pickle.dump(objs, f)


if __name__ == '__main__':
    create_statement_table()
    # # review_statement_table()
    # create_train_test_dataset()
    # review_function_statement()


    # process_func = ProcessFunc(debug=True)
    # test_code = process_func.rewrite_func(test_code)
    # print(test_code)
    # exit(0)

    # process_func = ProcessFunc(debug=False)
    # with open(os.path.join(poj104statement_dir, 'lstm-train.pkl'), 'rb') as f:
    #     objs = pickle.load(f)
    # for index,obj in enumerate(objs):
    #     # if index < 549:
    #     #     process_func.debug = True
    #     #     continue
    #     print(f'\n{index} function')
    #     code = obj['function']
    #     code = comment_remover(code)
    #     # print(code)
    #     code = process_func.rewrite_func(code)
