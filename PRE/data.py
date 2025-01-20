'''
    The implement of evaluated task data loader
'''

import os
import json, csv
import sys
from PRE.utils import read_from_tsv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

class DataLoader:
    '''
    The loader to load for evaluated task, with given prompt template to generate a series of prompts feeding for each LLM
    '''
    def __init__(self, args):
        self.path_data = args['path_data'] # the load path for the data
        self.format = args['format'] # the data format, csv (need a title line) or json (each line is a single data item)

        # 处理tsv数据为json
        if (args['format'] == 'tsv'):
            # lines = open(self.path_data).readlines()
            path_out = f"{self.path_data[:-4]}.json"
            
            if os.path.exists(path_out):
                pass
            else:
                fout = open(path_out, 'w')
                datas = read_from_tsv(self.path_data, ["content", "summary"])
                for data in datas[1:]:
                    text = data["content"].replace('\\n', '\n')
                    fout.write(json.dumps({"original_text": text}) + '\n')
            
            self.format = 'json'
            self.path_data = path_out

        self.path_prompt = args['path_prompt'] if 'path_prompt' in args else None # the path of prompt template. In the prompt template, using {{key}} for the replacement of the key. For example, in the prompt "You need answer a question: {{question}}", the "question" field need to be included in the data
        if not os.path.exists(self.path_data):
            raise FileExistsError("Load task data failed: file not exist!")
        assert self.format in ['csv', 'json']
        
    
    def generate_reader(self):
        if self.format == 'csv':
            with open(self.path_data, encoding='utf-8') as f:
                gen = csv.DictReader(f, skipinitialspace=True)
        elif self.format == 'json':
            gen = open(self.path_data, encoding='utf-8')
        else:
            raise Exception("Invalid data format")
        return gen
    
    def get_prompt(self):
        if self.path_prompt is None:
            raise Exception("Exception: missing argument path_prompt")
        if not os.path.exists(self.path_prompt):
            raise FileExistsError("Load task prompt template failed: file not exist!")
        self.template_prompt = open(self.path_prompt, encoding='utf-8').read().strip()
        
        gen = self.generate_reader()
        
        for row in gen:
            if self.format == 'json':
                item = json.loads(row.strip())
            else:
                item = row
            
            prompt = self.template_prompt
            for key in item:
                prompt = prompt.replace("{{" + key + "}}", item[key])
            yield prompt # a generator to return each prompt
    
    def get_task_items(self):
        data_list = []
        gen = self.generate_reader()
        for row in gen:
            if self.format == 'json':
                item = json.loads(row.strip())
            elif self.format == 'csv':
                item = dict(row)
            data_list.append(item)
        return data_list
