'''
    Implement some commonly used functions
'''


import re
import math
import os
import sys
import csv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas

def parse_num(res):
    all_nums = re.findall(r"-?\d+(?:\.\d+)?", res)  # 所有数字
    probs1 = re.findall(r"\b\d+(?:\.\d+)?/\d+\b", res) # 所有分数
    probs1_nums = re.finditer(r"\b(\d+(\.\d+)?)/\d+\b" , res) # 用于提取分子
    
    probs2 = re.findall(r"\b\d+(?:\.\d+)?\s+out\s+of\s+\d+\b", res) # 所有 out of
    probs2_nums = re.finditer(r"\b(\d+(\.\d+)?)\s+out\s+of\s+\d+\b" , res)

    if len(all_nums) == 0:
        print("this res doesn't have num! ", res)
        return -1

    answer = 0
    for match in probs1_nums:
        answer = match.group(1)

    for match in probs2_nums:
        answer = match.group(1)

    if answer == 0:
        for num in all_nums:
            if float(num) >= 1 and float(num) <= 5:  # 指定范围！！！
                answer = num

    if answer == 0:
        print("this res doesn't have right num! ", res)
        return -1

    if answer.find(".") != -1:
        print("this res doesn't have integer! idx ", i)
        return -1
    return answer

def parse_score_key(path, path_out, mode):
    response = get_response(path)
    # print(response)
    key_doubtful = ['doubtful', 'uncertain', 'moderate', 'confident', 'absolute']
    key_null = ['null', 'low', 'medium', 'high', 'expert']
    keys = key_doubtful if mode == 1 else key_null
    fout = open(path_out, 'w')
    for id, res in enumerate(response):
        res = res.lower()
        ans = ""
        ans_id = -1
        for i, key in enumerate(keys):
            if res.find(key) != -1:
                ans = key
                ans_id = i
        if (ans_id == -1):
            print(id)
            continue
        line = {"task_id": id, "scores": ans, "score": ans_id + 1}
        line = json.dumps(line)
        fout.write(line + '\n')

def parse_response(response, parse_type, nominal_list=None, nominal_ticks=None):
    '''
    parse_type: int, float or str
    if parse_type = str, then required parameter nominal_list and nominal_ticks
    nominal_list: a series of nominal types, its name
    nomianl_ticks: the corresponding nominal number (int)
    '''
    assert parse_type in ['int', 'float', 'str']
    if parse_type == 'int':
        return parse_num(response)
    elif parse_type == 'float':
        nums = re.findall(r"-?\d+\.?\d*", response)
        if len(nums) == 0:
            return None
        return int(nums[0])
    elif parse_type == 'str':
        appear_pos, cur_idx = -math.inf, -1
        response = response.lower()
        for idx, label in enumerate(nominal_list):
            pos = response.find(label.lower())
            if pos != -1: # really appear!
                if pos > appear_pos: # 选择最靠近结尾的匹配
                    appear_pos, cur_idx = pos, idx
        if cur_idx == -1:
            return 0
        else:
            return nominal_ticks[cur_idx]
    