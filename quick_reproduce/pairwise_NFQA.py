import json, math
import numpy as np
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
annotation_pointwise_path = base_dir + "/data-NFQA/annotation/pointwise.json"
annotation_preference_path = base_dir + "/data-NFQA/annotation/preference.json"
def get_median(arr):
    return list(sorted(arr))[2]

data = json.load(open(annotation_pointwise_path, 'r'))

prefers = json.load(open(annotation_preference_path, 'r'))

prefers_dict = {}
for key in prefers:
    item = key.split("%")
    prefers_dict[f"{item[0]}={item[1]}={item[2]}"] = prefers[key]
    prefers_dict.update({f"{item[0]}={item[2]}={item[1]}": [-ii for ii in prefers[key]]})

models = ['chatgpt', 'claude', 'ChatGLM2-6B', 'fastchat-t5-3b', 'RWKV-4-Raven-7B-v11', 'alpaca-7b', 'vicuna-7b']
models2idx = dict()
for i, m in enumerate(models):
    models2idx[m] = i

cnts = [0 for _ in range(5)]
for t in data:
    for m in models:
        ls = data[t][m]
        l = get_median(ls)

        cnts[l-1] += 1
# print(cnts)
ties = [0, 0] # tie, not tie
threshold = 5
for item in prefers_dict:
    ps = prefers_dict[item]
    v = 0
    for p in ps:
        if p < -threshold:
            v -= 1
        elif p > threshold:
            v += 1
    if abs(v) >= 2:
        ties[1] += 1
    else:
        ties[0] += 1
# print(ties)
# assert 1 == 0

res = []
texts = open(f'../data-NFQA/samples_NFCATS.tsv', 'r').readlines()
texts = [t.split('\t')[0].strip().replace('\\n', '\n') for t in texts]
for t in texts:
    items = [get_median(data[t][m]) for m in models]
    # print(items)
    res.append(items)

res = np.array(res)
print(res.shape)


# res是100个task，和每个task下7个model的中位数得分

_cnt = 0
for items in res:
    for i in range(len(items)):
        for j in range(i):
            if items[i] == items[j]:
                _cnt += 1
# print('tied number = ', _cnt)


pairs = [["ChatGLM2-6B", "alpaca-7b"], ["alpaca-7b", "ChatGLM2-6B"], ["chatgpt", "alpaca-7b"], ["alpaca-7b", "chatgpt"], ["chatgpt", "ChatGLM2-6B"], ["ChatGLM2-6B", "chatgpt"], ["claude", "alpaca-7b"], ["alpaca-7b", "claude"], ["claude", "ChatGLM2-6B"], ["ChatGLM2-6B", "claude"], ["claude", "chatgpt"], ["chatgpt", "claude"], ["fastchat-t5-3b", "alpaca-7b"], ["alpaca-7b", "fastchat-t5-3b"], ["fastchat-t5-3b", "ChatGLM2-6B"], ["ChatGLM2-6B", "fastchat-t5-3b"], ["fastchat-t5-3b", "chatgpt"], ["chatgpt", "fastchat-t5-3b"], ["fastchat-t5-3b", "claude"], ["claude", "fastchat-t5-3b"], ["RWKV-4-Raven-7B-v11", "alpaca-7b"], ["alpaca-7b", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "ChatGLM2-6B"], ["ChatGLM2-6B", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "chatgpt"], ["chatgpt", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "claude"], ["claude", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "fastchat-t5-3b"], ["fastchat-t5-3b", "RWKV-4-Raven-7B-v11"], ["vicuna-7b", "alpaca-7b"], ["alpaca-7b", "vicuna-7b"], ["vicuna-7b", "ChatGLM2-6B"], ["ChatGLM2-6B", "vicuna-7b"], ["vicuna-7b", "chatgpt"], ["chatgpt", "vicuna-7b"], ["vicuna-7b", "claude"], ["claude", "vicuna-7b"], ["vicuna-7b", "fastchat-t5-3b"], ["fastchat-t5-3b", "vicuna-7b"], ["vicuna-7b", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "vicuna-7b"]]

tied_used = True


aggs = dict()
answers_agg = dict() 

mode_key = "cps"

filter_mode = {
    "wo":{
        "evalutors": ['vicuna-7b', 'chatglm3-6b', 'baichuan2-13b', 'fastchat-t5-3b', 'chatgpt', 'chatglm_pro', 'gpt4'],
        "mode": 1,
        "weights": {

        }
    },
    "g":{
        "evalutors": ['fastchat-t5-3b', 'chatgpt', 'chatglm_pro', 'gpt4'],
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
            "chatglm_pro":0.7481,
            "chatgpt":0.5876,
            "gpt4":0.8686
        }   
    },
    "p":{
        "evalutors": ['fastchat-t5-3b', 'chatgpt', 'chatglm_pro', 'gpt4'],
        "mode": 1,
        "weights": {
            "fastchat-t5-3b":0.845,
            "chatglm_pro":0.9175,
            "chatgpt":0.8025,
            "gpt4":0.94
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
        "mode": 2,
        "weights": {
            "chatglm_pro":(0.9175+0.7481+1)/3,
            "chatgpt":(0.8025+0.5876+1)/3,
            "gpt4":(0.94+0.8686+1)/3
        }
    },
    "chatEval_two_turns":{
        "evalutors": ['chatEval_two_turns'],
        "mode": 1,
        "weights": {
        }
    },
    "chatEval":{
        "evalutors": ['chatEval'],
        "mode": 1,
        "weights": {
        }
    },
    "gpt4":{
        "evalutors": ['gpt4'],
        "mode": 1,
        "weights": {
        }
    },
    "chatgpt":{
        "evalutors": ['chatgpt'],
        "mode": 1,
        "weights": {
        }
    },
    "chatglm_pro":{
        "evalutors": ['chatglm_pro'],
        "mode": 1,
        "weights": {
        }
    }
}

weight_mode = filter_mode[mode_key]["mode"]
models_eval = filter_mode[mode_key]["evalutors"]
weights = filter_mode[mode_key]["weights"]

accurays = []
consis = []
models_filtered = []
ii_list = []
EXAM_THRESHOLD = 0.
CONS_THRESHOLD = 0.
REL_THRESHOLD = 0.8
total_vote = 0.
tag = 1

pair2acc = {}


def vote_weight(acc, model, mode=weight_mode):
    if mode == 1:
        return 1
    elif mode == 2:
        return acc
    elif mode == 3:
        return weights[model]
    else:
        return np.log(acc / (1-acc))

def get_res(responses, model):
    global total_vote
    def consistency_cal(responses):
        labels = []
        for i, line in enumerate(responses):
            label = 'others'
            l = line.lower()
            idxOne, idxTwo = l.find('one'), l.find('two')
            idxOne = np.inf if idxOne == -1 else idxOne
            idxTwo = np.inf if idxTwo == -1 else idxTwo
            if min(idxOne, idxTwo) != np.inf:
                if idxOne < idxTwo:
                    label = 'one'
                elif idxOne > idxTwo:
                    label = 'two'
            labels.append(label)
        _acc, _total = 0., 0.
        for i in range(len(labels)):
            if i % 200 < 100:
                l1, l2 = labels[i], labels[i+100]
                if l1 == 'others' or l2 == 'others':
                    _acc += .5
                elif l1 != l2:
                    _acc += 1.
                _total += 1.
        return _acc / _total


    def qualified_exam(models_exam=['chatgpt', 'fastchat-t5-3b', 'alpaca-7b']):
        _cnt_total, _cnt_correct = 0., 0.
        for i, line in enumerate(responses):
            ma, mb = pairs[i // 100]
            if ma not in models_exam or mb not in models_exam:
                continue
            ia, ib = models2idx[ma], models2idx[mb]
            i_task = i % 100

            label = 'others'
            l = line.lower()
            idxOne, idxTwo = l.find('one'), l.find('two')
            idxOne = np.inf if idxOne == -1 else idxOne
            idxTwo = np.inf if idxTwo == -1 else idxTwo
            if min(idxOne, idxTwo) != np.inf:
                if idxOne < idxTwo:
                    label = 'one'
                elif idxOne > idxTwo:
                    label = 'two'

            sa, sb = res[i_task, ia], res[i_task, ib]

            if sa == sb:
                key = f'{i_task}={ma}={mb}'
                ls_prf = prefers_dict[key]
                threshold = 5  # <= 5 -> tie
                vv = 0
                for l in ls_prf:
                    if l < -threshold:
                        vv -= 1
                    elif l > threshold:
                        vv += 1
                if vv >= 2:
                    sb += 0.5
                elif vv <= -2:
                    sa += 0.5
            if sa != sb:
                _cnt_total += 1.
                if label == 'one' and sa > sb:
                    _cnt_correct += 1.
                elif label == 'two' and sa < sb:
                    _cnt_correct += 1.
                elif label == 'others':
                    _cnt_correct += 0.5
        return _cnt_correct / _cnt_total

    acc_exam = qualified_exam()
    acc_cons = consistency_cal(responses)


    outputs_llm, outputs_llm2 = [], []
    weight_factor = acc_exam
    # weight_factor = 0
    print(f'model: {model}, weight = {vote_weight(weight_factor, model)}')

    total_vote += vote_weight(weight_factor, model) * 2 # two vote place
    cnt_correct, cnt_total = 0., 0.
    labels = []
    for i, line in enumerate(responses):
        ma, mb = pairs[i//100]
        ia, ib = models2idx[ma], models2idx[mb]
        i_task = i % 100

        label = 'others'
        l = line.lower()
        idxOne, idxTwo = l.find('one'), l.find('two')
        idxOne = np.inf if idxOne == -1 else idxOne
        idxTwo = np.inf if idxTwo == -1 else idxTwo
        if min(idxOne, idxTwo) != np.inf:
            if idxOne < idxTwo:
                label = 'one'
            elif idxOne > idxTwo:
                label = 'two'

        labels.append(label)

        sa, sb = res[i_task, ia], res[i_task, ib]

        if sa == sb and tied_used:
            key = f'{i_task}={ma}={mb}'
            ls_prf = prefers_dict[key]
            # print(ls_prf)
            threshold = 5 # <= 5 -> tie
            vv = 0
            for l in ls_prf:
                if l < -threshold:
                    vv -= 1
                elif l > threshold:
                    vv += 1
            if vv >= 2:
                sb += 0.5
            elif vv <= -2:
                sa += 0.5
        if sa != sb:
            score = 0.
            cnt_total += 1.
            if label == 'one' and sa > sb:
                cnt_correct += 1.
                score = 1.
            elif label == 'two' and sa < sb:
                cnt_correct += 1.
                score = 1.
            elif label == 'others':
                cnt_correct += 0.5
                score = 0.5

            ii = i - 100 if (i % 200) >= 100 else i
            if ii not in aggs:
                ii_list.append(ii)
                aggs[ii] = []
                if sa > sb:
                    item = [ia, ib]
                else:
                    item = [ib, ia]
                answers_agg[ii] = item

            aggs[ii].append(score*vote_weight(weight_factor, model))
            if i % 200 < 100:
                outputs_llm.append(score)
            else:
                outputs_llm2.append(score)

    # 将 cnt_correct / cnt_total 分开存
    if i // 100 not in pair2acc:
        pair2acc[i//100] = [cnt_correct, cnt_total]
    else:
        pair2acc[i//100][0] += cnt_correct
        pair2acc[i//100][1] += cnt_total

    cnt_cons = 0.
    ds_cons, ds_incons = [], []
    num = 0.
    total, tieL, tieS = 0., 0., 0.
    for i in range(len(labels)):
        if i % 200 < 100:

            ma, mb = pairs[i // 100]
            ia, ib = models2idx[ma], models2idx[mb]
            i_task = i % 100
            sa, sb = res[i_task, ia], res[i_task, ib]
            ds = abs(sa - sb)


            l1, l2 = labels[i], labels[i + 100]
            if l1 == 'others' or l2 == 'others':
                cnt_cons += 0.5
            elif l1 != l2:
                cnt_cons += 1.
                ds_cons.append(ds)
            else:
                ds_incons.append(ds)

            s = 0
            if l1 == 'one':
                s += 1.
            elif l1 == 'others':
                s += 0.5
            if l2 == 'two':
                s += 1.
            elif l2 == 'others':
                s += 0.5
            total += 1.
            if sa == sb:
                tieL += 1.
            if s == 1.:
                tieS += 1.
            if sa != sb and s != 1.:
                if (sa < sb and s < 1.) or (sa > sb and s > 1.):
                    num += 1.
                else:
                    num -= 1.
    tau = num / math.sqrt(total - tieL) / math.sqrt(total - tieS)
    print(f'model {model}, tau = {tau}')
    print()
    print(f'output llm, len = {len(outputs_llm)}, {len(outputs_llm2)}')
    print(cnt_correct / cnt_total)

    ratio_cons = cnt_cons * 2. / len(labels)

    accurays.append(cnt_correct / cnt_total)
    consis.append(ratio_cons)
    models_filtered.append(model)

    for i in range(len(outputs_llm)):
        outputs_llm[i] = (outputs_llm[i] + outputs_llm2[i]) / 2.

    os.makedirs('outputs-llm', exist_ok=True)

for iii, model in enumerate(models_eval):
    responses = open(f'{base_dir}/data-NFQA/evaluator_response_processed/response0-NFQA-100-onlyL_pairwise_processed_{model}.json', 'r').readlines()[:4200]
    get_res(responses, model)

print(f'accuray = {accurays}')
print(f'consis = {consis}')
print(f'model_filtered = {models_filtered}')

records_agg = []

outputs_llm = []
cnt_agg = 0.
scores = []
for i in ii_list:
    items = aggs[i]    # 每个task_id 每个待测答案对 的 score
    ms = answers_agg[i]
    ss = sum(items)
    l = 0.
    if ss * 2 > total_vote:
        scores.append(1)
        cnt_agg += 1.
        ms += [1., 0.]
        l = 1.
    elif ss * 2 == total_vote:
        scores.append(0.5)
        cnt_agg += 0.5
        ms += [.5, .5]
        l = .5
    else:
        scores.append(0)
        ms += [0., 1.]
    records_agg.append(ms)
    outputs_llm.append(l)
acc_agg = cnt_agg / float(len(aggs))
path_out = f"{base_dir}/data-NFQA/t_test/{mode_key}_pairwise.json"
f_out = open(path_out, "w")
line = {"scores": scores}
line = json.dumps(line)
f_out.write(line + "\n")
print(f'acc_agg = {acc_agg}')

