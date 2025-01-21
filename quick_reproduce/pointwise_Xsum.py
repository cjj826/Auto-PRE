import statistics
import json
import math
import os
from tqdm import tqdm
import csv
from scipy.stats import spearmanr, kendalltau
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
annotation_pointwise_path = base_dir + "/data-Xsum/annotation/pointwise.json"
annotation_preference_path = base_dir + "/data-Xsum/annotation/preference.json"

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

def cal(x, y):
    assert len(x) == len(y)
    cnt = 0
    for i,j in zip(x, y):
        if i * j == 0: cnt += 0.5
        elif i == j: cnt += 1
    return cnt / len(x) 

def get_ks(x, y):
    kendall, kendall_p_value = kendalltau(x, y, variant='c')
    if math.isnan(kendall):
        kendall = 0
    spearman, spearman_p_value = spearmanr(x, y)
    if math.isnan(spearman):
        spearman = 0
    return kendall, spearman

models = ["chatgpt", "claude", "ChatGLM2-6B", "fastchat-t5-3b", "RWKV-4-Raven-7B-v11", "alpaca-7b", "vicuna-7b"]

T = len(models) * (len(models) - 1) / 2

model_idx = {}

for i, model in enumerate(models):
    model_idx[model] = i

task_mturk2id = {}

with open(annotation_pointwise_path) as hm:
    fh = json.load(hm)
    datas = read_from_tsv(f"{base_dir}/data-Xsum/XSum-100-thre2500_v2.tsv", ["content", "summary"])

    for i, key in enumerate(fh):
        task_mturk2id[i] = -1
        key = key.replace("\n", "\\n")
        for id, data in enumerate(datas[1:]):
            text = data["content"]
            if text == key:
                task_mturk2id[i] = id
                break

tie_res = {}

with open(annotation_preference_path) as tie:
    tie_results = json.load(tie)
    for tie_result in tie_results:
        task_id = tie_result[0]
        x = model_idx[tie_result[1]]
        y = model_idx[tie_result[2]]
        A_better = 0; B_better = 0
        res = 0
        for score in tie_result[3]:
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
## acc best
# p 5-level w=1
# p 100-level w=4

## spearman best
# p 5-level w=1
# p 100-level w=4

mode_key = "p"
level = "100-level"

filter_mode = {
    "wo":{
        "evalutors": ['vicuna-7b', 'chatglm3-6b', 'baichuan2-13b', 'fastchat-t5', 'gpt-3.5-turbo', 'chatglm_pro', 'gpt-4'],
        # "evalutors": ['glm4'],
        "mode": 1,
        "weights": {

        }
    },
    "g":{
        "evalutors": ['chatglm3-6b','baichuan2-13b', 'fastchat-t5', 'gpt-3.5-turbo', 'chatglm_pro', 'gpt-4'],
        "mode": 2,
        "weights": {
        }
    },
    "c":{
        "evalutors": ['fastchat-t5', 'chatglm_pro', 'gpt-3.5-turbo', 'gpt-4'],
        "mode": 1,
        "weights": {
            "vicuna-7b":0.0002,
            "baichuan2-13b":0.4357,
            "chatglm3-6b":0.0264,
            "fastchat-t5":0.6662,
            "chatglm_pro":0.8360,
            "gpt-3.5-turbo":0.5948,
            "qianwen": 0.8247619047619048,
            "gpt-4":0.8952
        }   
    },
    "p":{
        "evalutors": ['chatglm3-6b', 'fastchat-t5', 'gpt-3.5-turbo', 'chatglm_pro', 'gpt-4'],
        "mode": 1,
        "weights": {
            'chatglm3-6b': 0.77,
            "fastchat-t5":0.865,
            "chatglm_pro":0.99,
            "gpt-3.5-turbo":0.955,
            "gpt-4":0.98,
            "qianwen": 0.985
        }
    },
    "s":{
        "evalutors": ['baichuan2-13b', 'fastchat-t5', 'gpt-3.5-turbo', 'chatglm_pro', 'gpt-4'],
        "mode": 1,
        "weights":{
            
        }
    },
    "cps":{
        "evalutors": ['fastchat-t5', 'chatglm_pro', 'gpt-3.5-turbo', 'gpt-4'],
        "mode": 2,
        "weights": {
            "fastchat-t5":(0.865+1+0.6662)/3,
            "chatglm_pro":(0.99+1+0.8360)/3,
            "gpt-3.5-turbo":(0.955+1+0.5948)/3,
            "gpt-4":(0.98+1+0.8952)/3,
            'qianwen': (0.8952+1+0.985)/3
        }
    },
    "chatEval": {
        "evalutors": ['chatEval'],
        "mode": 1,
        "weights": {
        }
    },
    "gpt-4": {
        "evalutors": ['gpt-4'],
        "mode": 1,
        "weights": {
        }
    }
}

base_paths = {
    "100-level": base_dir + "/data-Xsum/100-level/Xsum_result_detail_",
    "5-level": base_dir + "/data-Xsum/5-level/Xsum_result_detail_"
}

base_path = base_paths[level]

def qualified_exam(model, models_exam=['chatgpt', 'fastchat-t5-3b', 'alpaca-7b']):
    if mode_key != "g":
        return
    model_result_path = base_path + model + ".json"
    # read the human annotations
    with open(annotation_pointwise_path) as f:
        annotations = json.load(f)
        with open(model_result_path) as f2:
            model_results = json.load(f2)
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
                        resultA_y = model_results[modelA][task_id]
                        resultB_y = model_results[modelB][task_id]

                        if resultA_y != -1 and resultB_y != -1 and resultA_y != resultB_y:
                            res_y = -1 if resultA_y > resultB_y else 1
                        
                        result_of_x.append(res_x)
                        result_of_y.append(res_y)

            total_acc = cal(result_of_x, result_of_y)
            weights[model] = total_acc
            print("model is ", model, "and the exam acc is ", total_acc)

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
        return np.log(weights[model] / (1-weights[model]))
    elif weight_mode == 4:
        return math.exp(weights[model]/ (1 - weights[model]))

def get_kendall_res(model):
    # read the human annotations
    model_result_path = base_path + model + ".json"
    with open(annotation_pointwise_path) as f:
        annotations = json.load(f)
        with open(model_result_path) as f2:
            model_results = json.load(f2)
            # the first loop: task
            result_of_x = []
            result_of_y = []
            total_acc = 0
            task_x = []
            task_y = []
            k = 0
            s = 0
            nan_t = 0
            c = 0 # 一致对数
            d = 0 # 不一致对数
            tx = 0 # x值不变，人工认为一样好的数量
            ty = 0 # y值不变， 模型认为一样好的数量
            cnt = 0
            for ooid, key in tqdm(enumerate(annotations)):
                annotation = annotations[key]
                task_id = task_mturk2id[ooid] # get the true task_id
                task_x = []
                task_y = []

                for m in models:
                    key = m + "_" + str(task_id)
                    if key not in total_results.keys():
                        total_results[key] = model_results[m][task_id] * vote_weight(model)
                    else:
                        total_results[key] += model_results[m][task_id] * vote_weight(model)
                    
                for i in range(len(models)):
                    modelA = models[i]
                    result_x = statistics.median(annotation[modelA])
                    result_y = model_results[modelA][task_id]
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
                                print("the task id is {task_id} and the modelA is {modelA}, the modelB is {modelB}\n")
                                res_x = 0
                            else:
                                res_x = tie_res[key]
                                if res_x == 0:
                                    continue
                        else:
                            res_x = -1 if resultA_x > resultB_x else 1
                        
                        res_y = 0
                        
                        resultA_y = model_results[modelA][task_id]
                        resultB_y = model_results[modelB][task_id]

                        # if resultA_y != resultB_y:
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
                # if math.isnan(t_k) or math.isnan(t_s):
                #     nan_t += 1
                #     continue
                k += t_k
                s += t_s

                
            total_acc = cal(result_of_x, result_of_y)
            print(f"final spearman is {s/(100 - nan_t):.4f}")
            print("the final acc is ", total_acc)

if __name__ == '__main__':

    path_out = f"./significant_test/Xsum/{mode_key}_{level}.json"
    
    for model in evalutors:
        qualified_exam(model)   # 是否通过筛选
        get_kendall_res(model)

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
        nan_t = 0
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
            annotation = annotations[key]
            task_id = task_mturk2id[ooid] # get the true task_id
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
            if math.isnan(t_k) or math.isnan(t_s):
                # nan_t += 1
                continue
            k += t_k
            s += t_s

        # kendall = (c - d) / math.sqrt((cnt - tx) * (cnt - ty))
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
        print(f"final spearman is {s/(100 - nan_t):.4f}")
        # print("the agg kendall is ", kendall)
        print("the agg acc is ", total_acc)