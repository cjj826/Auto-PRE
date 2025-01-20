import statistics
import json
import math
import os
from tqdm import tqdm
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
annotation_pointwise_path = base_dir + "/data-NFQA/annotation/pointwise.json"
annotation_preference_path = base_dir + "/data-NFQA/annotation/preference.json"


def cal(x, y):
    assert len(x) == len(y)
    cnt = 0
    for i,j in zip(x, y):
        if i * j == 0: cnt += 0.5
        elif i == j: cnt += 1
    return cnt / len(x) 

from scipy.stats import kendalltau, spearmanr

def get_ks(x, y):
    kendall, kendall_p_value = kendalltau(x, y, variant='c')
    if math.isnan(kendall):
        kendall = 0
    spearman, spearman_p_value = spearmanr(x, y)
    if math.isnan(spearman):
        spearman = 0
    return kendall, spearman

models = ["alpaca-7b", "ChatGLM2-6B", "chatgpt", "claude", "fastchat-t5-3b", "RWKV-4-Raven-7B-v11", "vicuna-7b"]

T = len(models) * (len(models) - 1) / 2

model_idx = {}

for i, model in enumerate(models):
    model_idx[model] = i

task_mturk2id = {}

with open(annotation_pointwise_path) as hm:
    fh = json.load(hm)
    datas = open("../data-NFQA/samples_NFCATS.tsv").readlines()

    for i, key in enumerate(fh):
        task_mturk2id[i] = -1

        for id, data in enumerate(datas):
            text = data.strip()
            if text == key:
                task_mturk2id[i] = id
                break
        if task_mturk2id[i] == -1:
            print(i)

tie_res = {}

with open(annotation_preference_path) as tie:
    tie_results = json.load(tie)
    for item in tie_results:
        task_id, model1, model2 = item.split("%")
        x = model_idx[model1]
        y = model_idx[model2]
        A_better = 0; B_better = 0
        res = 0
        for score in tie_results[item]:
            if score >= -30 and score <= -6:
                A_better += 1
            elif score >= 6 and score <= 30:
                B_better += 1
            else: 
                pass
        if A_better - B_better >= 2: # A比B好是-1，B比A好是1
            res = -1
        elif B_better - A_better >= 2:
            res = 1
        else:
            res = 0
        if x > y:
            t = y
            y = x
            x = t
            res = -res
        key = str(task_id) + "-" + str(x) + "-" + str(y)
        tie_res[key] = res

mode_key = "cps"
level = "5-level"

## acc
# p 5-level w=1
# cps 100-level w=1

## spearman
# cps 5-level w=1
# cps 100-level w=1

filter_mode = {
    "wo":{
        "evalutors": ['vicuna-7b', 'chatglm3-6b', 'baichuan2-13b', 'fastchat-t5-3b', 'chatgpt', 'chatglm_pro', 'gpt4'],
        "mode": 1,
        "weights": {

        }
    },
    "g":{
        "evalutors": ['chatglm_pro', 'gpt4'],
        "mode": 2,
        "weights": {
        }
    },
    "c":{
        "evalutors": ['chatgpt', 'chatglm_pro', 'gpt4'],
        "mode": 1,
        "weights": {
            "vicuna-7b":0.4414,
            "baichuan2-13b":0.0595,
            "chatglm3-6b":0.3914,
            "fastchat-t5-3b":0.5160,
            "chatglm_pro":0.5876,
            "chatgpt":0.7481,
            "gpt4":0.8686,
            "qianwen":0.8366666666666667
        }   
    },
    "p":{
        "evalutors": ['fastchat-t5-3b', 'chatgpt', 'chatglm_pro', 'gpt4'],
        "mode": 4,
        "weights": {
            "fastchat-t5-3b":0.8475,
            "chatglm_pro":0.8,
            "chatgpt":0.725,
            "gpt4":0.8525,
            'qianwen': 0.925
        }
    },
    "s":{
        "evalutors": ['baichuan2-13b', 'fastchat-t5-3b', 'chatglm_pro', 'gpt4'],
        "mode": 1,
        "weights":{
            
        }
    },
    "cps":{
        "evalutors": ['chatglm_pro', 'gpt4'],
        "mode": 1,
        "weights": {
            "chatglm_pro":(0.5876+1+0.8)/3,
            "chatgpt":(0.7481+1+0.725)/3,
            "gpt4":(0.8686+1+0.8525)/3,
            'qianwen':(0.8366666666666667 + 0.925 + 1)/3,
        }
    },
    "gpt4": {
        "evalutors": ['gpt4'],
        "mode": 1,
         "weights": {
        }
    },
    "chatEval": {
        "evalutors": ['chatEval'],
        "mode": 1,
        "weights": {
        }
    },
}

base_paths = {
    "100-level": base_dir + "/data-NFQA/evaluator_response_processed/responses1-NFQA-100_pointwise_100level_processed_",
    "5-level": base_dir + "/data-NFQA/evaluator_response_processed/responses1-NFQA-100_pointwise_5level_processed_"
}

base_path = base_paths[level]

import re

def parse_answer(model_results):
    scores = []
    for line in model_results:
        nums = re.findall(r"-?\d+", line)
        if len(nums) == 0:
            scores.append(-1)
        else:
            scores.append(int(nums[0]))
    return scores


def qualified_exam(model, models_exam=['chatgpt', 'fastchat-t5-3b-3b', 'alpaca-7b']):
    if mode_key != "g":
        return
    model_result_path = base_path + model + ".json"
    # read the human annotations
    with open(annotation_pointwise_path) as f:
        annotations = json.load(f)
        with open(model_result_path) as f2:
            model_results = parse_answer(f2.readlines())
            # the first loop: task
            result_of_x = []
            result_of_y = []
            for ooid, key in tqdm(enumerate(annotations)):
                task_id = task_mturk2id[ooid] # get the true task_id
                annotation = annotations[key]

                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        res_x = 0 # -1 表示 A 好，1 表示 B 好，0 表示认为一样好
                        modelA = models[i]; modelB = models[j]
                        if modelA not in models_exam or modelB not in models_exam:
                            continue
                        # print("the pair is ", modelA, " and ", modelB)
                        resultA_x = statistics.median(annotation[modelA])
                        resultB_x = statistics.median(annotation[modelB])
                        if resultA_x == resultB_x:
                            key = str(task_id) + "-" + str(i) + "-" + str(j)
                            if key not in tie_res.keys():
                                print("the task id is {task_id} and the modelA is {modelA}, the modelB is {modelB}\n")
                                res_x = 0
                            else:
                                res_x = tie_res[key]
                                if res_x == 0:
                                    continue
                        else:
                            res_x = -1 if resultA_x > resultB_x else 1
                        
                        res_y = 0
                        resultA_y = model_results[i*100+task_id]
                        resultB_y = model_results[j*100+task_id]

                        if resultA_y != -1 and resultB_y != -1 and resultA_y != resultB_y:
                            res_y = -1 if resultA_y > resultB_y else 1
                        
                        result_of_x.append(res_x)
                        result_of_y.append(res_y)

            total_acc = cal(result_of_x, result_of_y)
            weights[model] = total_acc
            print("the exam acc is ", total_acc)

# 索引答案：evaluator + model + taskId
# 对于融合答案：model + taskId

total_results = {}

weight_mode = filter_mode[mode_key]["mode"]
evalutors = filter_mode[mode_key]["evalutors"]
weights = filter_mode[mode_key]["weights"]

import numpy as np
def vote_weight(model):
    if weight_mode == 1:
        return 1
    elif weight_mode == 2:
        return weights[model]
    elif weight_mode == 3:
        return math.exp(weights[model] / (1 - weights[model]))
    else:
        return np.log(weights[model] / (1-weights[model]))

def get_kendall_res(model):
    # read the human annotations
    model_result_path = base_path + model + ".json"
    with open(annotation_pointwise_path) as f:
        annotations = json.load(f)
        with open(model_result_path) as f2:
            model_results = parse_answer(f2.readlines())
            # the first loop: task
            kendall = {}
            right_num = 0
            ACC = {}
            task_x = []
            task_y = []
            k = 0
            s = 0
            result_of_x = []
            result_of_y = []
            total_acc = 0
            ans = 0
            c = 0 # 一致对数
            d = 0 # 不一致对数
            tx = 0 # x值不变，人工认为一样好的数量
            ty = 0 # y值不变， 模型认为一样好的数量
            cnt = 0
            for ooid, key in tqdm(enumerate(annotations)):
                task_id = task_mturk2id[ooid] # get the true task_id
                task_x = []
                task_y = []
                annotation = annotations[key]

                for i, m in enumerate(models):
                    key = m + "_" + str(task_id)
                    # 非法值时不参与融合
                    if model_results[i*100+task_id] == -1:
                        continue
                    if key not in total_results.keys():
                        total_results[key] = model_results[i*100+task_id] * vote_weight(model)
                    else:
                        total_results[key] += model_results[i*100+task_id] * vote_weight(model)

                for i in range(len(models)):
                    modelA = models[i]
                    result_x = statistics.median(annotation[modelA])
                    result_y = model_results[i*100+task_id]
                    task_x.append(result_x)
                    task_y.append(result_y)

                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        res_x = 0 # -1 表示 A 好，1 表示 B 好，0 表示认为一样好
                        modelA = models[i]; modelB = models[j]
                        # print("the pair is ", modelA, " and ", modelB)
                        resultA_x = statistics.median(annotation[modelA])
                        resultB_x = statistics.median(annotation[modelB])
                        if resultA_x == resultB_x:
                            key = str(task_id) + "-" + str(i) + "-" + str(j)
                            if key not in tie_res.keys():
                                print(annotation[modelA])
                                print(annotation[modelB])
                                print(task_id, modelA, modelB, resultA_x, resultB_x)
                                # print("the task id is {task_id} and the modelA is {modelA}, the modelB is {modelB}\n")
                                return 
                            else:
                                res_x = tie_res[key]
                                if res_x == 0:
                                    continue
                        else:
                            res_x = -1 if resultA_x > resultB_x else 1
                        
                        res_y = 0
                        resultA_y = model_results[i*100+task_id]
                        resultB_y = model_results[j*100+task_id]

                        if resultA_y != -1 and resultB_y != -1 and resultA_y != resultB_y:
                            res_y = -1 if resultA_y > resultB_y else 1
                        
                        result_of_x.append(res_x)
                        result_of_y.append(res_y)

                t_k, t_s = get_ks(task_x, task_y)
                k += t_k
                s += t_s
                
            total_acc = cal(result_of_x, result_of_y)
            print(f"final spearman is {s/100:.4f}")
            print("the final acc is ", total_acc)
        
if __name__ == '__main__':
    
    for model in evalutors:
        qualified_exam(model)   # 是否通过筛选
        get_kendall_res(model)
    
    path_out = f"./significant_test/NF_CATS/{mode_key}_{level}.json"
    
    with open(annotation_pointwise_path) as f:
        annotations = json.load(f)
        # the first loop: task
        kendall = {}
        right_num = 0
        ACC = {}
        result_of_x = []
        result_of_y = []
        task_x = []
        task_y = []
        k = 0
        s = 0
        total_acc = 0
        ans = 0
        c = 0 # 一致对数
        d = 0 # 不一致对数
        tx = 0 # x值不变，人工认为一样好的数量
        ty = 0 # y值不变， 模型认为一样好的数量
        cnt = 0
        for ooid, key in tqdm(enumerate(annotations)):
            task_id = task_mturk2id[ooid] # get the true task_id
            annotation = annotations[key]
            task_x = []
            task_y = []

            for i in range(len(models)):
                    modelA = models[i]
                    result_x = statistics.median(annotation[modelA])
                    result_y = total_results[modelA + "_" + str(task_id)]
                    task_x.append(result_x)
                    task_y.append(result_y)

            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    res_x = 0 # -1 表示 A 好，1 表示 B 好，0 表示认为一样好
                    modelA = models[i]; modelB = models[j]
                    # print("the pair is ", modelA, " and ", modelB)
                    resultA_x = statistics.median(annotation[modelA])
                    resultB_x = statistics.median(annotation[modelB])
                    if resultA_x == resultB_x:
                        key = str(task_id) + "-" + str(i) + "-" + str(j)
                        if key not in tie_res.keys():
                            print(f"the task id is {task_id} and the modelA is {modelA}, the modelB is {modelB}\n")
                            # f_out.write(f'the task id is {task_id} and the modelA is {modelA}, the modelB is {modelB}\n')
                            res_x = 0
                        else:
                            res_x = tie_res[key]
                            if res_x == 0:
                                continue
                    else:
                        res_x = -1 if resultA_x > resultB_x else 1
                    
                    res_y = 0
                    key = modelA + "_" + str(task_id)
                    resultA_y = total_results[key]
                    key = modelB + "_" + str(task_id)
                    resultB_y = total_results[key]

                    if resultA_y != -1 and resultB_y != -1 and resultA_y != resultB_y:
                        res_y = -1 if resultA_y > resultB_y else 1
                    
                    result_of_x.append(res_x)
                    result_of_y.append(res_y)

                    if res_x == 0 or res_y == 0: # 对于不变对
                        tx += (res_x == 0)
                        ty += (res_y == 0)
                    elif res_x != res_y:
                        d += 1
                    else:
                        c += 1
                    cnt += 1
            t_k, t_s = get_ks(task_x, task_y)
            k += t_k
            s += t_s

        total_acc = cal(result_of_x, result_of_y)
        
        scores = []
        for i, j in zip(result_of_x, result_of_y):
            if i * j == 0: scores.append(0.5)
            elif i == j: scores.append(1)
            else: scores.append(0)
        f_out = open(path_out, "w")
        line = {"scores": scores}
        line = json.dumps(line)
        f_out.write(line + "\n")

        print(f"final spearman is {s/100:.4f}")
        print("the agg acc is ", total_acc)