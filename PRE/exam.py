'''
    The implement of the qualified exam module
'''

import os
import yaml
import warnings
import json
import sys
import math
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from PRE.data import DataLoader
from PRE.api import Auto_API
from PRE.utils import parse_response
from PRE.utils import read_from_tsv
import statistics

from tqdm import tqdm
import numpy as np

class EXAM:
    '''
    Conduct qualified exam, filtering qualified LLMs to become peer reviewers
    '''
    def __init__(self, args) -> None:
        self.exam = args["exam"]
        self.dev = args["dev"]
        self.source = args['source'] # same or others; same: the evaluated task and responses, others: independent prompts, no need for refer item
        self.mode = args['mode'] # pointwise, pairwise
        self.parser_type = args['parser_type'] # int, float, str
        '''
        If the source is same,
        In pointwise mode, the data consists key "#index" (the line index of the task) and key "#source" (the LLM to generate the response). The expected evaulate response is an integer or float number;
        In pairwise mode, the data consists key "#index" (the line index of the task), key "#source1" (the LLM 1 to generate the response) and key "#source2" (the LLM 2 to generate the response). The expected evaluate response is three possible token, meaning -1 (1 is better), 0 (tied), 1 (2 is better) respectively
        also, if we conduct reference exam, for each exam data item, it requires key "#answer" denotes the gold standard (integer for the pairwise mode)
        '''
        assert self.source in ['same', 'others']
        assert self.mode in ['pointwise', 'pairwise']
        assert self.parser_type in ['int', 'float', 'str']
        if self.parser_type == 'str':
            self.nominal_list = ['one', 'tie', 'two']
            self.nominal_ticks = [-1, 0, 1]
        else:
            self.nominal_list, self.nominal_ticks = None, None
        
        if self.source == 'same': # load generated task data and responses
            path_config_task_data = args['config_task_data']
            self.task_name = args['task_name']
            self.save_dir = args['save_dir'] # the exam result save dir, the exam evaluation save filename = [save_dir] / exam_responses / [task_name]_[model_name].json, each line is one result with json {response: str, result: float/int}
            if not os.path.exists(path_config_task_data):
                raise FileExistsError("Load task_data config failed: file not exist!")

            config_task = yaml.load(open(path_config_task_data, 'r'), Loader=yaml.FullLoader) # single task config
            data_loader = DataLoader(config_task) # a task data loader
            self.task_data = data_loader.get_task_items()
            self.path_exam_same_data = args['path_exam_same_data']
            self.format_exam_same_data = args['format_exam_same_data']
        else: # load other exam data
            self.path_exam_others_data = args['path_exam_others_data']
            self.format_exam_others_data = args['format_exam_others_data']
            if not os.path.exists(self.path_exam_others_data):
                raise FileExistsError("Load exam others mode data failed: file not exist!")
        self.reference_exam = args['conduct_reference_exam'] # True or False, whether to compare the responses v.s. gold standard
        self.inner_consistency_exam = args['conduct_inner_consistency_exam'] # True or False, whether to conduct inner-consistency exam
        if self.mode == 'pairwise':
            if self.reference_exam:
                self.p_gold = float(args['p_gold']) if 'p_gold' in args else 0.6 # accuarcy v.s. gold standard
            if self.inner_consistency_exam:
                self.p_cons = float(args['p_cons']) if 'p_cons' in args else 0.6 # consistency between two kinds of prompts
        elif self.mode == 'pointwise':
            self.metric_pointwise = args['metric_pointwise'] if 'metric_pointwise' in args else 'EM' # EM (exact match, proportion  >= threshold) or MSE (mean square error, mse <= threshold)
            assert self.metric_pointwise in ['EM', "MSE"]
            if self.reference_exam:
                if self.metric_pointwise == 'EM':
                    self.p_gold = float(args['p_gold']) if 'p_gold' in args else 0.6 # accuarcy v.s. gold standard
                elif self.metric_pointwise == 'MSE':
                    self.MSE_acc = float(args['MSE_gold']) if 'MSE_gold' in args else 1. # MSE v.s. gold standard
                
            if self.inner_consistency_exam:
                if self.metric_pointwise == 'EM':
                    self.p_cons = float(args['p_cons']) if 'p_cons' in args else 0.6 # consistency between two kinds of prompts
                elif self.metric_pointwise == 'MSE':
                    self.MSE_cons = float(args['MSE_cons']) if 'MSE_cons' in args else 1. # MSE between two kinds of prompts
    
    def load_exam_prompts(self, prompt_template):
        if self.source == 'others':
            loader = DataLoader({"path_data": self.path_exam_others_data,
                                 "format": self.format_exam_others_data,})
            data_others = loader.get_task_items()
            prompts = []
            for item in data_others:
                prompt = prompt_template
                for key in item:
                    prompt = prompt.replace("{{" + key + "}}", item[key])
                prompts.append(prompt)
            if self.reference_exam:
                answers = [item['#answer'] for item in data_others]
            else:
                answers = None
            return prompts, answers
        elif self.source == 'same':
            loader = DataLoader({"path_data": self.path_exam_same_data,
                                 "format": self.format_exam_same_data,})
            samples_same = loader.get_task_items()
            evaluatees_list = set()
            if self.mode == 'pointwise':
                for sample in samples_same:
                    evaluatees_list.add(sample['#source'])
            elif self.mode == 'pairwise':
                for sample in samples_same:
                    evaluatees_list.add(sample['#source1'])
                    evaluatees_list.add(sample['#source2'])
            responses_evaluatee_dict = dict()
            for ev in evaluatees_list:
                responses = [] # responses list for evaluatee ev
                path = f"{self.save_dir}/task_responses/{self.task_name}_{ev}.json"
                if not os.path.exists(path):
                    raise FileExistsError(f"Load {path} failed: file not exist!")
                with open(path, 'r') as f:
                    while True:
                        line = f.readline().strip()
                        if line:
                            response = json.loads(line)['response']
                            responses.append(response)
                        else:
                            break
                responses_evaluatee_dict[ev] = responses
            
            prompts = []
            for sample in samples_same:
                sidx = sample['#index']
                task = dict(self.task_data[sidx])
                if self.mode == 'pointwise':
                    src = sample['#source']
                    task['#source'] = responses_evaluatee_dict[src][sidx]
                elif self.mode == 'pairwise':
                    src1 = sample['#source1']
                    src2 = sample['#source2']
                    task['#source1'] = responses_evaluatee_dict[src1][sidx]
                    task['#source2'] = responses_evaluatee_dict[src2][sidx]
                prompt = prompt_template
                for key in task:
                    prompt = prompt.replace("{{" + key + "}}", task[key])
                prompts.append(prompt)
            
            if self.reference_exam:
                answers = [item['#answer'] for item in samples_same]
            else:
                answers = None
            return prompts, answers
    
    def calculate_metric(self, resultsA, resultsB) -> float: 
        '''
        Calculate the evaluation metric between resultsA and resultsB
        pointwise or pairwise; EM/accuary or MSE (minus)
        '''
        assert len(resultsA) == len(resultsB)
        assert len(resultsA) > 0
        N = len(resultsA)
        p = 0.
        
        for j in range(N):
            r, a = resultsA[j], resultsB[j]
            if r * a > 0:
                p += 1.
            elif r * a == 0:
                p += .5

        p /= float(N)
        return p
        
    def conduct_exam(self, config_api_evaluator):
        '''
        Conduct qualified exam, return a list of qualified apis with the same format of list config_api_evaluator, and their scores [score_list (refer acc, inner acc) for each qualified LLM], MSE will put the minus one
        '''
        if self.dev:
            if self.exam == "pertinence":
                # param：template_prompt, p_thre = mean, L1='ChatGLM2-6B', L2='gpt-3.5-turbo', Q, Q' (Q由gpt4改造得到)

                self.output_limit = 1
                template_prompt = open(f"{base_dir}/data/prompt_{self.task_name}_pairwise{self.output_limit}.txt", encoding='utf-8').read().strip()
                return self.pertinence_exam(config_api_evaluator, template_prompt, L1="ChatGLM2-6B", L2="gpt-3.5-turbo")
            elif self.exam == "self_confidence":
                # param: template_prompt, easy_set, hard_set
                self.get_confidence_method = "prob" # direct / prob
                template_prompt = ""
                if self.get_confidence_method == "direct":
                    self.nominal_list = ['null', 'low', 'medium', 'high', 'expert']
                    self.nominal_ticks = [1, 2, 3, 4, 5]
                    self.output_limit = "null"
                    template_prompt = open(f"{base_dir}/data/prompt_key_null_{self.task_name}.txt", encoding='utf-8').read().strip()
                elif self.get_confidence_method == "prob":
                    self.output_limit = 1
                    template_prompt = open(f"{base_dir}/data/prompt_{self.task_name}_pairwise{self.output_limit}.txt", encoding='utf-8').read().strip()
                return self.self_confidence_exam(config_api_evaluator, template_prompt)
            elif self.exam == "consistency":
                # param: template_prompt='', models_exam
                self.nominal_list = ['one', 'tie', 'two']
                self.nominal_ticks = [-1, 0, 1]
                template_prompt = open(f"{base_dir}/data/prompt_eval_pairwise2.txt", encoding='utf-8').read().strip()
                return self.consistency_exam(config_api_evaluator, template_prompt, models_exam=['alpaca-7b','baichuan2-13b']) 
            elif self.exam == "human":
                # param: template_prompt='', p_thre = 0.6, models_exam 
                annotation_path = f'{base_dir}/data-{self.task_name}/annotation/pointwise.json'
                pre_path = f'{base_dir}/data-{self.task_name}/annotation/preference.json'
                if self.mode == "pointwise":
                    template_prompt = open(f"{base_dir}/data/prompt_eval_5level.txt", encoding='utf-8').read().strip()
                    return self.qualified_exam(config_api_evaluator, template_prompt, p_thre = 0.6, models_exam=['chatgpt', 'fastchat-t5-3b', 'alpaca-7b'], annotation_path = annotation_path, pre_path = pre_path)
                elif self.mode == "pairwise":
                    template_prompt = open(f"{base_dir}/data/prompt_eval_pairwise1.txt", encoding='utf-8').read().strip()
                    return self.qualified_exam(config_api_evaluator, template_prompt, p_thre = 0.6, models_exam=['chatgpt', 'fastchat-t5-3b', 'alpaca-7b'], annotation_path = annotation_path, pre_path = pre_path)
                return [], []
        else:
            apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_api_evaluator]
            prompts, answers = self.load_exam_prompts(self.template_prompt)
            if self.inner_consistency_exam:
                prompts2, answers2 = self.load_exam_prompts(self.template_prompt2)
            
            os.makedirs(f"{self.save_dir}/exam_responses", exist_ok=True)
            qualified_apis, scores_qualified = [], [] # configs of these qualified apis, its corresponding api
            for i, api in enumerate(apis):
                path_out = f"{self.save_dir}/exam_responses/{self.task_name}_{api.model_name}.json"

                if os.path.exists(path_out):
                    data = open(path_out).readlines()
                else:
                    data = []
                if len(data) < len(prompts):
                    fout = open(path_out, 'w')
                    for line in data:
                        fout.write(line)
                    for prompt in prompts[len(data):]:
                        response_orig = api.chat(prompt)
                        result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                        line = json.dumps({"response": response_orig,
                                            'result': result_parse})
                        data.append(line)
                        fout.write(line + '\n')
                    fout.close()
                results = [json.loads(line.strip())['result'] for line in data]
                
                eval_this = [config_api_evaluator[i]]
                
                if self.reference_exam:
                    p_refer = self.calculate_metric(results, answers)
                    p_thre = None
                    if self.mode == 'pairwise':
                        p_thre = self.p_gold
                    elif self.mode == 'pointwise':
                        if self.metric_pointwise == 'EM':
                            p_thre = self.p_gold
                        elif self.metric_pointwise == 'MSE':
                            p_thre = -self.MSE_acc
                    
                    if p_refer < p_thre:
                        print(f'model {api.model_name} failed to pass the reference exam')
                        continue
                    eval_this.append(p_refer)
                
                if self.inner_consistency_exam:
                    path_out = f"{self.save_dir}/exam_responses/{self.task_name}_{api.model_name}__prompt2.json"

                    if os.path.exists(path_out):
                        data = open(path_out).readlines()
                    else:
                        data = []
                    if len(data) < len(prompts2):
                        fout = open(path_out, 'w')
                        for line in data:
                            fout.write(line)
                        for prompt in prompts2[len(data):]:
                            response_orig = api.chat(prompt)
                            result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                            line = json.dumps({"response": response_orig,
                                                'result': result_parse})
                            data.append(line)
                            fout.write(line + '\n')
                        fout.close()
                    results2 = [json.loads(line.strip())['result'] for line in data]

                    p_inner = self.calculate_metric(results, results2)
                    p_thre = None
                    if self.mode == 'pairwise':
                        p_thre = self.p_cons
                    elif self.mode == 'pointwise':
                        if self.metric_pointwise == 'EM':
                            p_thre = self.p_cons
                        elif self.metric_pointwise == 'MSE':
                            p_thre = -self.MSE_cons
                    
                    if p_inner < p_thre:
                        print(f'model {api.model_name} failed to pass the inner-consistency exam')
                        continue
                    eval_this.append(p_inner)
                
                qualified_apis.append(config_api_evaluator[i])
                scores_qualified.append(eval_this)
            return qualified_apis, scores_qualified
    
    def get_confidence(self, api, prompts, mode):
        os.makedirs(f"{self.save_dir}/exam_self_confidence/{self.get_confidence_method}", exist_ok=True)
        path_out = f"{self.save_dir}/exam_self_confidence/{self.get_confidence_method}/{self.task_name}_{api.model_name}_{mode}_{self.output_limit}.json"

        if os.path.exists(path_out):
            data = open(path_out).readlines()
        else:
            data = []
        if len(data) < len(prompts):
            fout = open(path_out, 'w')
            for line in data:
                fout.write(line)
            cnt = len(data)
            for idx, prompt in tqdm(enumerate(prompts[cnt:])):
                print(idx)
                if self.get_confidence_method == "direct":
                    response_orig = api.chat(prompt)
                    result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                    line = json.dumps({"response": response_orig, 'result': result_parse, "prompt": prompt})
                elif self.get_confidence_method == "prob":
                    # 另一个接口
                    response_orig, res_pro, probability_of_one, probability_of_two, top3 = api.chat_prob(prompt)
                    line = json.dumps({"response": response_orig, "res_pro": res_pro, "probability_of_one": probability_of_one, "probability_of_two": probability_of_two, "top3": top3, "prompt": prompt})
                data.append(line)
                fout.write(line + '\n')
            fout.close()
        if self.get_confidence_method == "prob":
            results = []
            for line in data:
                res_pro = float(json.loads(line.strip())['res_pro'])
                if api.model_name == 'gpt-3.5-turbo':
                    res_pro = res_pro / 100
                if res_pro > 0: # 没有输出特定token
                    results.append(-math.log(res_pro))
        elif self.get_confidence_method == "direct":
            results = [json.loads(line.strip())['result'] for line in data]
        for i in range(len(results)):
            if results[i] == None:
                results[i] = 0
        # print(np.mean(results), np.std(results))
        return np.mean(results)
                
    def self_confidence_exam(self, config_api_evaluator, prompt_template='', easy=['gpt4', 'RWKV-4-Raven-7B-v11'], hard=['gpt4', 'claude']):
        os.makedirs(f"{self.save_dir}/exam_self_confidence", exist_ok=True)
        print("loading evaluatee answer sets...")
        responses_evaluatee_dict = dict()
        for set in [easy, hard]:
            for ev in set:
                path = f"{self.save_dir}/task_responses/{self.task_name}_{ev}.json"
                responses = self.load_response_from_file(path)
                responses_evaluatee_dict[ev] = responses
        
        print("loading easy prompts...")
        easy_prompts = []
        for idx in tqdm(range(len(self.task_data))):
            L1_response = responses_evaluatee_dict[easy[0]][idx]
            L2_response = responses_evaluatee_dict[easy[1]][idx]

            task = dict(self.task_data[idx])
            task['#source1'] = L1_response
            task['#source2'] = L2_response
            prompt = prompt_template
            for key in task:
                prompt = prompt.replace("{{" + key + "}}", task[key])
            easy_prompts.append(prompt)
            # transfer the order
            task = dict(self.task_data[idx])
            task['#source1'] = L2_response
            task['#source2'] = L1_response
            prompt = prompt_template
            for key in task:
                prompt = prompt.replace("{{" + key + "}}", task[key])
            easy_prompts.append(prompt)
    
        print("loading hard prompts...")
        hard_prompts = []
        for idx in tqdm(range(len(self.task_data))):
            L1_response = responses_evaluatee_dict[hard[0]][idx]
            L2_response = responses_evaluatee_dict[hard[1]][idx]

            task = dict(self.task_data[idx])
            task['#source1'] = L1_response
            task['#source2'] = L2_response
            prompt = prompt_template
            for key in task:
                prompt = prompt.replace("{{" + key + "}}", task[key])
            hard_prompts.append(prompt)
            # transfer the order
            task = dict(self.task_data[idx])
            task['#source1'] = L2_response
            task['#source2'] = L1_response
            prompt = prompt_template
            for key in task:
                prompt = prompt.replace("{{" + key + "}}", task[key])
            hard_prompts.append(prompt)

        print("calculate...")
        apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_api_evaluator]
        qualified_apis = []
        scores_qualified = []

        for i, api in enumerate(apis):
            print(api.model_name)
            easy_mean_confidence = self.get_confidence(api, easy_prompts, "easy")
            print("easy mean confidence: ", easy_mean_confidence)
            hard_mean_confidence = self.get_confidence(api, hard_prompts, "hard")
            print("hard mean confidence: ", hard_mean_confidence)
            eval_this = [config_api_evaluator[i]]
            eval_this.append(1)
            if self.get_confidence_method == "prob":
                if easy_mean_confidence < hard_mean_confidence:
                    qualified_apis.append(config_api_evaluator[i])
                    scores_qualified.append(eval_this)
            elif self.get_confidence_method == "direct":
                if easy_mean_confidence > hard_mean_confidence:
                    qualified_apis.append(config_api_evaluator[i])
                    scores_qualified.append(eval_this)
        return qualified_apis, scores_qualified
    
    def load_response_from_file(self, path):
        # 从file中读取，返回数组形式
        print("loading from ", path)
        responses = []
        if not os.path.exists(path):
            raise FileExistsError(f"Load {path} failed: file not exist!")
        with open(path, 'r') as f:
            while True:
                line = f.readline().strip()
                if line:
                    response = json.loads(line)['response']
                    responses.append(response)
                else:
                    break
        return responses
    
    def pertinence_exam(self, config_api_evaluator, prompt_template='', L1='ChatGLM2-6B', L2="gpt-3.5-turbo"):
        os.makedirs(f"{self.save_dir}/exam_pertinence", exist_ok=True)
        print("loading L1 responses...")
        path = f"{self.save_dir}/task_responses/{self.task_name}_{L1}.json"
        L1_responses = self.load_response_from_file(path)[:100]

        print("loading Q' by gpt4 change...")
        # Q' in data-Xsum/new_text.json
        # change_Q = "###Task: Rewrite the given input to generate a new input, requiring the new input to be similar to the original input form, but with different key information.\n###Given input: {{original_text}}\n###new input:"
        # Q' in data-NFQA/new_text.json
        change_Q = "###Task: Rewrite the given input to generate a new input, requiring that:\n1. the new input is similar to the original input form but has different key information.\n2.the correct answer to the new input should not be a correct answer to the given input.\n###Given input: {{original_text}}\n###new input:"
        
        gpt4_api = Auto_API.init_api_by_name("gpt-4")
        prompts = []
        for idx in tqdm(range(len(self.task_data))):
            task = self.task_data[idx]
            prompt = change_Q
            prompt = prompt.replace("{{original_text}}", task["original_text"])
            prompts.append(prompt)
        path_out = f"{base_dir}/data-{self.task_name}/new_text.json"
        if os.path.exists(path_out):
            data = open(path_out).readlines()
        else:
            data = []
        if len(data) < len(prompts):
            fout = open(path_out, 'w')
            for line in data:
                fout.write(line)
            cnt = len(data)
            for idx, prompt in enumerate(prompts[cnt:]):
                response_orig = gpt4_api.chat(prompt)
                line = json.dumps({"new_text": response_orig})
                data.append(line)
                fout.write(line + '\n')
            fout.close()
        new_texts = [json.loads(line.strip())['new_text'] for line in data]
        print("loading L2 responses...")
        # L2 is recorded in data/L2_responses/model_name.json
        prompts = []
        if self.task_name == "Xsum":
            get_L2 = "Task: Generate a short summary of the text in at most 64 words.\n\n Text:{{new_text}}"
        elif self.task_name == "NFQA":
            get_L2 = "Task: Please answer the following question within 200 words.\n\nQuestion:{{new_text}}\n\nAnswer:"
        for new_text in new_texts:
            prompt = get_L2.replace("{{new_text}}", new_text)
            prompts.append(prompt)
        if os.path.exists(path_out):
            data = open(path_out).readlines()
        else:
            data = []
        os.makedirs(f"{base_dir}/data-{self.task_name}/L2_responses", exist_ok=True)
        path_out = f"{base_dir}/data-{self.task_name}/L2_responses/{L2}.json"
        # 由L2得到对应的api调用
        L2_api = Auto_API.init_api_by_name(L2)
        if os.path.exists(path_out):
            data = open(path_out).readlines()
        else:
            data = []
        if len(data) < len(prompts):
            fout = open(path_out, 'w')
            for line in data:
                fout.write(line)
            cnt = len(data)
            for idx, prompt in enumerate(prompts[cnt:]):
                response_orig = L2_api.chat(prompt)
                line = json.dumps({"response": response_orig})
                data.append(line)
                fout.write(line + '\n')
            fout.close()
        L2_responses = [json.loads(line.strip())['response'] for line in data]
        print("finish collect L2 responses...")
        print("get all prompts and answers...")
        prompts = []; answers = []
        for idx in tqdm(range(len(self.task_data))):
            L1_response = L1_responses[idx]
            L2_response = L2_responses[idx]

            task = dict(self.task_data[idx])
            task['#source1'] = L1_response
            task['#source2'] = L2_response
            prompt = prompt_template
            for key in task:
                prompt = prompt.replace("{{" + key + "}}", task[key])
            prompts.append(prompt)
            answers.append(-1)
            # transfer the order
            task = dict(self.task_data[idx])
            task['#source1'] = L2_response
            task['#source2'] = L1_response
            prompt = prompt_template
            for key in task:
                prompt = prompt.replace("{{" + key + "}}", task[key])
            prompts.append(prompt)
            answers.append(1)

        print("calculate metric...")
        apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_api_evaluator]
        qualified_apis = []
        scores_qualified = []

        for i, api in enumerate(apis):
            os.makedirs(f"{self.save_dir}/exam_pertinence/{L1}_L1-{L2}_L2-{self.output_limit}", exist_ok=True)
            path_out = f"{self.save_dir}/exam_pertinence/{L1}_L1-{L2}_L2-{self.output_limit}/{self.task_name}_{api.model_name}.json"
            results = self.get_results_from_prompts(api, prompts, path_out)
            eval_this = [config_api_evaluator[i]]
            acc = self.calculate_metric(results, answers)
            eval_this.append(acc)
            print(api.model_name, acc)
            qualified_apis.append(config_api_evaluator[i])
            scores_qualified.append(eval_this)

        # threshold : mean
        average_score = 0
        for score in scores_qualified:
            score = score[1]
            average_score += score
        average_score /= len(scores_qualified)
        print(f"average_score: {average_score}")
        pass_qualified_apis = []
        pass_scores_qualified = []
        for idx, score in enumerate(scores_qualified):
            if score[1] > average_score:
                pass_qualified_apis.append(qualified_apis[idx])
                pass_scores_qualified.append(scores_qualified[idx])

        return pass_qualified_apis, pass_scores_qualified

    def get_results_from_prompts(self, api, prompts, path_out):
        # 针对一组prompts，得到api对应的输出
        if os.path.exists(path_out):
            data = open(path_out).readlines()
        else:
            data = []
        if len(data) < len(prompts):
            fout = open(path_out, 'w')
            for line in data:
                fout.write(line)
            cnt = len(data)
            for idx, prompt in enumerate(prompts[cnt:]):
                response_orig = api.chat(prompt)
                result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                line = json.dumps({"response": response_orig, 'result': result_parse, "prompt": prompt})
                data.append(line)
                fout.write(line + '\n')
            fout.close()
        results = [json.loads(line.strip())['result'] for line in data]
        return results

    def consistency_exam(self, config_api_evaluator, prompt_template='', p_thre = 0.55, models_exam=['alpaca-7b','baichuan2-13b','chatglm_pro','ChatGLM2-6B','chatgpt','claude','fastchat-t5-3b','gpt4','lamma2-70b','RWKV-4-Raven-7B-v11','vicuna-7b']):
        os.makedirs(f"{self.save_dir}/exam_consistency", exist_ok=True)
        print("loading evaluatee answer sets...")
        responses_evaluatee_dict = dict()
        for ev in models_exam:
            responses = [] # responses list for evaluatee ev
            path = f"{self.save_dir}/task_responses/{self.task_name}_{ev}.json"
            responses = self.load_response_from_file(path)
            responses_evaluatee_dict[ev] = responses

        print("loading prompts...")
        before_prompts = []
        after_prompts = []
        for idx in tqdm(range(len(self.task_data))):
            for model_Aidx in range(0, len(models_exam)):
                for model_Bidx in range(model_Aidx + 1, len(models_exam)):
                    L1_response = responses_evaluatee_dict[models_exam[model_Aidx]][idx]
                    L2_response = responses_evaluatee_dict[models_exam[model_Bidx]][idx]
                    task = dict(self.task_data[idx])
                    task['#source1'] = L1_response
                    task['#source2'] = L2_response
                    prompt = prompt_template
                    for key in task:
                        prompt = prompt.replace("{{" + key + "}}", task[key])
                    before_prompts.append(prompt)
                    # transfer the order
                    task = dict(self.task_data[idx])
                    task['#source1'] = L2_response
                    task['#source2'] = L1_response
                    prompt = prompt_template
                    for key in task:
                        prompt = prompt.replace("{{" + key + "}}", task[key])
                    after_prompts.append(prompt)

        print("calculate metric...")
        apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_api_evaluator]
        qualified_apis = []
        scores_qualified = []

        for i, api in enumerate(apis):
            before_results = self.get_results_from_prompts(api, before_prompts, f"{self.save_dir}/exam_consistency/{self.task_name}_{api.model_name}_before.json")
            after_results = self.get_results_from_prompts(api, after_prompts, f"{self.save_dir}/exam_consistency/{self.task_name}_{api.model_name}_after.json")
            # mind that! reverse after_results
            after_results = [-res for res in after_results]
            eval_this = [config_api_evaluator[i]]
            acc = self.calculate_metric(before_results, after_results)
            print(api.model_name, acc)
            eval_this.append(acc)
            if acc > p_thre:
                qualified_apis.append(config_api_evaluator[i])
                scores_qualified.append(eval_this)
        
        # threshold : mean
        average_score = 0
        for score in scores_qualified:
            score = score[1]
            average_score += score
        average_score /= len(scores_qualified)
        print(f"average_score: {average_score}")
        pass_qualified_apis = []
        pass_scores_qualified = []
        for idx, score in enumerate(scores_qualified):
            if score[1] > average_score:
                pass_qualified_apis.append(qualified_apis[idx])
                pass_scores_qualified.append(scores_qualified[idx])

        return pass_qualified_apis, pass_scores_qualified
    
    
    def get_model_results_from_prompts(self, api, prompts, path_out):
        if os.path.exists(path_out):
            data = open(path_out).readlines()
        else:
            data = []
        if len(data) < len(prompts):
            fout = open(path_out, 'w')
            for line in data:
                fout.write(line)
            cnt = len(data)
            for idx, prompt in enumerate(prompts[cnt:]):
                response_orig = api.chat(prompt)
                result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                line = json.dumps({"response": response_orig, 'result': result_parse, "prompt": prompt})
                data.append(line)
                fout.write(line + '\n')
            fout.close()

        model_results = {}
        for line in data:
            l = json.loads(line.strip())
            l["result"] = -1 if l["result"] == None else l["result"] ###当值非法时，处理为-1 or 0
            if l["model"] not in model_results.keys():
                model_results[l["model"]] = [l["result"]]
            else:
                model_results[l["model"]].append(l["result"])
        return model_results

    def compare_between_human_and_model_results(self, annotation_path, pre_path, models_exam, model_results):
        task_mturk2id = {}

        with open(annotation_path) as hm:
            fh = json.load(hm)
            datas = read_from_tsv("../data/XSum-100-thre2500_v2.tsv", ["content", "summary"])

            for i, key in enumerate(fh):
                task_mturk2id[i] = -1
                key = key.replace("\n", "\\n")
                for id, data in enumerate(datas[1:]):
                    text = data["content"]
                    if text == key:
                        task_mturk2id[i] = id
                        break

        tie_res = {}
        models = ["chatgpt", "claude", "ChatGLM2-6B", "fastchat-t5-3b", "RWKV-4-Raven-7B-v11",
                "alpaca-7b", "vicuna-7b"]
        model_idx = {}
        for i, model in enumerate(models):
            model_idx[model] = i

        with open(pre_path) as tie:
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
    
        # read the human annotations
        with open(annotation_path) as f:
            annotations = json.load(f)
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
                                # f_out.write(f'the task id is {task_id} and the modelA is {modelA}, the modelB is {modelB}\n')
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


            # kendall = (c - d) / math.sqrt((cnt - tx) * (cnt - ty))
            acc = self.calculate_metric(result_of_x, result_of_y)
            return acc

    def get_median(self, arr):
        return list(sorted(arr))[2]

    def qualified_exam(self, config_api_evaluator, prompt_template = '', p_thre = 0.6, models_exam=['chatgpt', 'fastchat-t5-3b', 'alpaca-7b'], annotation_path = '', pre_path = ''):
        os.makedirs(f"{self.save_dir}/exam_human", exist_ok=True)
        print("loading evaluatee answer sets...")
        responses_evaluatee_dict = dict()
        for ev in models_exam:
            responses = [] # responses list for evaluatee ev
            path = f"{self.save_dir}/task_responses/{self.task_name}_{ev}.json"
            responses = self.load_response_from_file(path)
            responses_evaluatee_dict[ev] = responses
        
        if self.mode == "pointwise":
            print("loading prompts...")
            prompts = []
            for idx in tqdm(range(len(self.task_data))):
                for model in models_exam:
                    response = responses_evaluatee_dict[model][idx]
                    task = dict(self.task_data[idx])
                    task['#source'] = response
                    prompt = prompt_template
                    for key in task:
                        prompt = prompt.replace("{{" + key + "}}", task[key])
                    prompts.append(prompt)
            print("calculate metric...")
            qualified_apis = []
            scores_qualified = []
            apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_api_evaluator]
            for i, api in enumerate(apis):
                path_out = f"{self.save_dir}/exam_human/{self.task_name}_{api.model_name}.json"
                model_results = self.get_model_results_from_prompts(api, prompts, path_out)
                #  每一个大模型有一个model_results，这个model_results以被测大模型为key，每个value是一个数组，记录了taskid下的结果
                acc = self.compare_between_human_and_model_results(annotation_path, pre_path, models_exam, model_results)
                eval_this = [config_api_evaluator[i]]
                eval_this.append(acc)
                print(api.model_name, acc)
                if acc > p_thre:
                    qualified_apis.append(config_api_evaluator[i])
                    scores_qualified.append(eval_this)
            return qualified_apis, scores_qualified
        elif self.mode == "pairwise":
            if self.dev:
                # 前提：已经得到了该大模型对 4200 个待测大模型对的输出
                print("get responses...")
                qualified_apis = []
                scores_qualified = []
                apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_api_evaluator]
                for i, api in enumerate(apis):
                    responses = open(f'../data-NFQA/evaluator_response_processed/response0-NFQA-100-onlyL_pairwise_processed_{api.model_name}.json', 'r').readlines()[:4200]
                    acc = self.accuracy_withtie(annotation_path, pre_path, models_exam, responses)
                    eval_this = [config_api_evaluator[i]]
                    eval_this.append(acc)
                    print(api.model_name, acc)
                    if acc > p_thre:
                        qualified_apis.append(config_api_evaluator[i])
                        scores_qualified.append(eval_this)
                return qualified_apis, scores_qualified

            else:
                print("loading prompts...")
                print("get responses...")
                print("calculate metric...")
    
    def accuracy_withtie(self, annotation_path, pre_path, models_exam, responses):
        _cnt_total, _cnt_correct = 0., 0.
        pairs = [["ChatGLM2-6B", "alpaca-7b"], ["alpaca-7b", "ChatGLM2-6B"], ["chatgpt", "alpaca-7b"], ["alpaca-7b", "chatgpt"], ["chatgpt", "ChatGLM2-6B"], ["ChatGLM2-6B", "chatgpt"], ["claude", "alpaca-7b"], ["alpaca-7b", "claude"], ["claude", "ChatGLM2-6B"], ["ChatGLM2-6B", "claude"], ["claude", "chatgpt"], ["chatgpt", "claude"], ["fastchat-t5-3b", "alpaca-7b"], ["alpaca-7b", "fastchat-t5-3b"], ["fastchat-t5-3b", "ChatGLM2-6B"], ["ChatGLM2-6B", "fastchat-t5-3b"], ["fastchat-t5-3b", "chatgpt"], ["chatgpt", "fastchat-t5-3b"], ["fastchat-t5-3b", "claude"], ["claude", "fastchat-t5-3b"], ["RWKV-4-Raven-7B-v11", "alpaca-7b"], ["alpaca-7b", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "ChatGLM2-6B"], ["ChatGLM2-6B", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "chatgpt"], ["chatgpt", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "claude"], ["claude", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "fastchat-t5-3b"], ["fastchat-t5-3b", "RWKV-4-Raven-7B-v11"], ["vicuna-7b", "alpaca-7b"], ["alpaca-7b", "vicuna-7b"], ["vicuna-7b", "ChatGLM2-6B"], ["ChatGLM2-6B", "vicuna-7b"], ["vicuna-7b", "chatgpt"], ["chatgpt", "vicuna-7b"], ["vicuna-7b", "claude"], ["claude", "vicuna-7b"], ["vicuna-7b", "fastchat-t5-3b"], ["fastchat-t5-3b", "vicuna-7b"], ["vicuna-7b", "RWKV-4-Raven-7B-v11"], ["RWKV-4-Raven-7B-v11", "vicuna-7b"]]
        models = ['chatgpt', 'claude', 'ChatGLM2-6B', 'fastchat-t5-3b', 'RWKV-4-Raven-7B-v11', 'alpaca-7b', 'vicuna-7b']
        data = json.load(open(annotation_path, 'r'))
        prefers = json.load(open(pre_path, 'r'))
        prefers_dict = {}
        for key in prefers:
            item = key.split("%")
            prefers_dict[f"{item[0]}={item[1]}={item[2]}"] = prefers[key]
            prefers_dict.update({f"{item[0]}={item[2]}={item[1]}": [-ii for ii in prefers[key]]})
        models2idx = dict()
        res = []
        texts = open(f'../data-NFQA/samples_NFCATS.tsv', 'r').readlines()
        texts = [t.split('\t')[0].strip().replace('\\n', '\n') for t in texts]
        for t in texts:
            items = [self.get_median(data[t][m]) for m in models]
            res.append(items)

        res = np.array(res)

        # res是100个task，和每个task下7个model的中位数得分
        _cnt = 0
        for items in res:
            for i in range(len(items)):
                for j in range(i):
                    if items[i] == items[j]:
                        _cnt += 1

        for i, m in enumerate(models):
            models2idx[m] = i

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
    
    
