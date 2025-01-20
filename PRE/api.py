'''
    The implement of various LLM APIs
'''

from abc import ABC, abstractmethod
import traceback, time
import requests
import json
import math
import openai
import zhipuai
from openai import OpenAI
import httpx
# openai.api_base = "https://api.chatgptid.net/v1"
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import heapq

def print_top3(tokenizer, p_list):
    # 使用heapq找到前三大的数
    top_three = heapq.nlargest(3, p_list)
    # 找到这些数的索引
    top_three_indices = [(tokenizer.decode(p_list.index(x)), x) for x in top_three]
    return top_three_indices


class LLM_API(ABC):
    '''
    The base class for all kinds of LLM APIs
    '''
    @abstractmethod
    def __init__(self, args) -> None:
        self.model_name = args['model_name'] # model name
        self.api_type = None
    
    @abstractmethod
    def chat(self, prompt) -> str:
        '''
        The unified method for chatting with LLM, feeding the prompt, and then output the response
        '''
        raise NotImplementedError
    
    @staticmethod
    def instantiate_api(config):
        '''
        instantiate an LLM_API subclass object
        '''
        raise NotImplementedError


class OPENAI_API(LLM_API):
    '''
    OpenAI format API, please refer https://platform.openai.com/docs/api-reference/introduction for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "openai"
        if 'api_key' not in args:
            raise Exception("Exception: missing openai API argument api_key")
        self.api_key = args['api_key']
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5

    def request(self, messages) -> str:
        client = OpenAI(
            base_url="https://svip.xty.app/v1", 
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url="https://svip.xty.app/v1",
                follow_redirects=True,
            ),
        )

        for _ in range(self.max_tries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    temperature = 0.0,
                    top_p = 0.0,
                    seed = 24,
                    messages=messages,
                )

                if response:
                    answer = response.choices[0].message.content.strip()
                    print(answer)
                    return response
            except:
                traceback.print_exc()
                time.sleep(5)
                continue


    def chat(self, prompt) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt}) 
        response = self.request(messages)
        result = response.choices[0].message.content.strip()
        return result
    
    def request_prob(self, messages) -> str:
        client = OpenAI(
            base_url="https://svip.xty.app/v1", 
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url="https://svip.xty.app/v1",
                follow_redirects=True,
            ),
        )

        for _ in range(self.max_tries):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature = 0.0,
                    top_p = 0.0,
                    seed = 24,
                    messages=messages,
                    logprobs=True,
                    top_logprobs=3,
                )

                if response != None:
                    if response.choices[0].logprobs == None:
                        continue
                    answer = response.choices[0].message.content.strip()
                    print(answer)
                    return response
            except:
                traceback.print_exc()
                time.sleep(5)
                continue


    def chat_prob(self, prompt) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt}) 
        response = self.request_prob(messages)
        print(response)
        if response == None:
            return 'null', -1, -1, -1, []
        result = response.choices[0].message.content.strip()
        res_pro = -1
        probability_of_one = -1; probability_of_two = -1; top3 = []
        assert(response.choices[0].logprobs != None)
        if response.choices[0].logprobs:
            for content in response.choices[0].logprobs.content:
                # 输出 one / two 时，记录 top3
                res = content.top_logprobs[0].token.lower().strip()
                if res == "one" or res == "two":
                    top3 = []
                    probability_of_one = -1
                    probability_of_two = -1
                    res_pro = np.round(np.exp(content.top_logprobs[0].logprob)*100,6)
                    for token in content.top_logprobs:
                        s = token.token.strip()
                        p = np.round(np.exp(token.logprob)*100,6)
                        top3.append((s, p))
                        if s == "one": 
                            probability_of_one = p
                        elif s == "two":
                            probability_of_two = p
                    print(res_pro, probability_of_one, probability_of_two, top3)
                
        return result, res_pro, probability_of_one, probability_of_two, top3    

class FASTCHAT_API(LLM_API):
    '''
    FASTCHAT_API defined by myself
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "fastchat"
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        self.model_name_or_path = args['model_path']
        self.init = True
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                inputs = inputs.to('cuda:0')
                response = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature = 0.0)[0]
                response = self.tokenizer.decode(response.cpu()[0], skip_special_tokens=True)
                print(response)
                return response
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None
    
    def chat_prob(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                inputs = inputs.to('cuda:0')
                response, probabilities = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature = 0.0)
                res_pro = -1
                probability_of_one = -1; probability_of_two = -1; top3 = []
                for p in probabilities:
                    next_tokens = torch.argmax(p, dim=-1)
                    now_tokens = self.tokenizer.decode(next_tokens)
                    if now_tokens.lower() == "one" or now_tokens.lower() == "two":
                        res_pro = torch.max(p, dim=-1).values.item()
                        p_list = p.cpu().tolist()[0]
                        probability_of_one = p_list[self.tokenizer("one").input_ids[-1]]
                        probability_of_two = p_list[self.tokenizer("two").input_ids[-1]]
                        top3 = print_top3(self.tokenizer, p_list)
                        print(res_pro, probability_of_one, probability_of_two, top3)

                response = self.tokenizer.decode(response.cpu()[0], skip_special_tokens=True)
                print(response)
                return response, res_pro, probability_of_one, probability_of_two, top3
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

class GLM3_6B_API(LLM_API):
    '''
    GLM3_6B_API defined by myself
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "chatglm3-6b"
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        self.model_name_or_path = args['model_path']
        self.init = True
        self.count = 0
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                response, re_history, probabilities = self.model.chat(self.tokenizer, prompt, history=[], do_sample=False)
                print(response)
                return response
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

    def chat_prob(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                response, re_history, probabilities = self.model.chat(self.tokenizer, prompt, history=[], do_sample=False)
                res_pro = -1
                probability_of_one = -1; probability_of_two = -1; top3 = []
                for p in probabilities:
                    next_tokens = torch.argmax(p, dim=-1)
                    now_tokens = self.tokenizer.decode(next_tokens)
                    if now_tokens.lower() == "one" or now_tokens.lower() == "two":
                        res_pro = torch.max(p, dim=-1).values.item()
                        p_list = p.cpu().tolist()[0]
                        probability_of_one = p_list[self.tokenizer("one").input_ids[-1]]
                        probability_of_two = p_list[self.tokenizer("two").input_ids[-1]]
                        top3 = print_top3(self.tokenizer, p_list)
                        print(res_pro, probability_of_one, probability_of_two, top3)
                print(response)
                return response, res_pro, probability_of_one, probability_of_two, top3
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

class BAICHUAN_API(LLM_API):
    '''
    BAICHUAN_API defined by myself
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "baichuan2-13b"
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        self.model_name_or_path = args['model_path']
        self.init = True
        self.count = 0
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                messages = []
                messages.append({"role": "user", "content": prompt})
                response, probabilities = self.model.chat(self.tokenizer, messages)
                print(response)
                return response
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None
    
    def chat_prob(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                messages = []
                messages.append({"role": "user", "content": prompt})
                response, probabilities = self.model.chat(self.tokenizer, messages)
                res_pro = -1
                probability_of_one = -1; probability_of_two = -1; top3 = []
                for p in probabilities:
                    next_tokens = torch.argmax(p, dim=-1)
                    now_tokens = self.tokenizer.decode(next_tokens)
                    if now_tokens.lower() == "one" or now_tokens.lower() == "two":
                        res_pro = torch.max(p, dim=-1).values.item()
                        p_list = p.cpu().tolist()[0]
                        probability_of_one = p_list[self.tokenizer("one").input_ids[-1]]
                        probability_of_two = p_list[self.tokenizer("two").input_ids[-1]]
                        top3 = print_top3(self.tokenizer, p_list)
                        print(res_pro, probability_of_one, probability_of_two, top3)
                print(response)
                return response, res_pro, probability_of_one, probability_of_two, top3
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

class CLAUDE_API(LLM_API):
    '''
    Claude-1 API with slack engine, we adopt claude2openai package for implement. please refer https://github.com/ChuxiJ/claude2openai for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "claude"
        self.api_key = args['api_key']
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        self.MAX_LEN = int(args['MAX_LEN']) if 'MAX_LEN' in args else 3960
        if 'slack_api_token' not in args:
            raise Exception("Exception: missing Claude API argument slack_api_token")
        if 'bot_id' not in args:
            raise Exception("Exception: missing Claude API argument bot_id")
        if 'channel_id' not in args:
            raise Exception("Exception: missing Claude API argument channel_id")
        self.slack_api_token = args['slack_api_token']
        self.bot_id = args['bot_id']
        self.channel_id = args['channel_id']
    
    def chat(self, prompt) -> str:
        import claude2openai
        claude2openai.slack_api_token = self.slack_api_token
        claude2openai.bot_id = self.bot_id
        claude2openai.channel_id = self.channel_id
        for _ in range(self.max_tries):
            try:
                prompt = prompt[:self.MAX_LEN]
                chat_completion = claude2openai.ChatCompletion.create(model="claude", messages=[{"role": "user", "content": prompt}])
                return chat_completion.choices[0].message.content
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None


class BAIDU_API(LLM_API):
    '''
    Baidu API with slack engine, please refer https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2 for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "baidu"
        if 'token' not in args:
            raise Exception("Exception: missing Baidu API argument token")
        self.token = args['token']
        self.url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model_name}?access_token={self.token}"
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            try:
                payload = json.dumps({
                    "messages": [
                        {"role": "user", "content": prompt[:4800]}
                    ]
                })
                headers = {
                    'Content-Type': 'application/json'
                }

                response = requests.request("POST", self.url, headers=headers, data=payload)
                result = json.loads(response.text)['result']
                return result
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

class GLM_API(LLM_API):
    '''
    GLM API with slack engine, please refer https://open.bigmodel.cn/dev/api for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "glm"
        if 'api_key' not in args:
            raise Exception("Exception: missing GLM API argument token")
        self.api_key = args['api_key']
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5

    def request(self, messages) -> str:
        response = zhipuai.model_api.invoke(
            do_sample = False,
            temperature = 0.0,
            model="chatglm_pro",
            prompt=messages
        )
        if response['success']:
            if 'choices' not in response['data']:
                result = response['data']['outputText']
            else:
                result = response['data']['choices'][0]['content'].strip('"').strip().replace('\n', '\\n')
        else:
            result = response['msg']
        print(result)
        return result
    
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            try:
                zhipuai.api_key = self.api_key
                messages = []
                messages.append({"role": "user", "content": prompt})                
                response = self.request(messages)
                return response
            except:
                traceback.print_exc()
                time.sleep(8)
                continue
        return None
    
class VICUNA_API(LLM_API):
    '''
    VICUNA_API defined by myself
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "vicuna-7b"
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        self.model_name_or_path = args['model_path']
        self.init = True
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                inputs = inputs.to('cuda:0')
                response = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature = 0.0)[0]
                response = self.tokenizer.decode(response.cpu()[0], skip_special_tokens=True)[len(prompt):]
                print(response)
                return response
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

    def chat_prob(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                inputs = inputs.to('cuda:0')
                response, probabilities = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature = 0.0)
                res_pro = -1
                probability_of_one = -1; probability_of_two = -1; top3 = []
                for p in probabilities:
                    next_tokens = torch.argmax(p, dim=-1)
                    now_tokens = self.tokenizer.decode(next_tokens)
                    if now_tokens.lower() == "one" or now_tokens.lower() == "two":
                        res_pro = torch.max(p, dim=-1).values.item()
                        p_list = p.cpu().tolist()[0]
                        probability_of_one = p_list[self.tokenizer("one").input_ids[-1]]
                        probability_of_two = p_list[self.tokenizer("two").input_ids[-1]]
                        top3 = print_top3(self.tokenizer, p_list)
                        print(res_pro, probability_of_one, probability_of_two, top3)

                response = self.tokenizer.decode(response.cpu()[0], skip_special_tokens=True)
                print(response)
                return response, res_pro, probability_of_one, probability_of_two, top3
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

class ALPACA_API(LLM_API):
    '''
    ALPACA_API defined by myself
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "alpaca-7b"
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        self.model_name_or_path = args['model_path']
        self.init = True
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            if self.init:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    use_fast=True,
                    add_bos_token=False,
                    add_eos_token=False,
                    padding_side="left",
                )
                self.init = False
            try:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                inputs = inputs.to('cuda:0')
                response = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature = 0.0)[0]
                response = self.tokenizer.decode(response.cpu()[0], skip_special_tokens=True)[len(prompt):]
                print(response)
                return response
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None


# each item: [api_type, LLM_API child class name]
API_type2class_list = [['openai', OPENAI_API], ['claude', CLAUDE_API],
                       ['baidu', BAIDU_API], ['glm', GLM_API], ['fastchat', FASTCHAT_API], ['chatglm3-6b', GLM3_6B_API], 
                       ['baichuan2-13b', BAICHUAN_API], ['vicuna-7b', VICUNA_API], ['alpaca-7b', ALPACA_API]] 

class Auto_API:
    @staticmethod
    def instantiate_api(api_type, args) -> LLM_API:
        for at, _API in API_type2class_list:
            if api_type == at:
                return _API(args)
        raise Exception(f"Invalid api_type: {api_type}")
    def init_api_by_name(api_name) -> LLM_API:
        if api_name == "gpt-4":
            return OPENAI_API({'model_name': 'gpt-4', 'api_type': 'openai', 'api_key': 'your_api_key', 'max_tries': 5})
        elif api_name == "gpt-3.5-turbo":
            return OPENAI_API({'model_name': 'gpt-3.5-turbo', 'api_type': 'openai', 'api_key': 'your_api_key', 'max_tries': 5})
        elif api_name == "baichuan2-13b":
            return BAICHUAN_API({'model_name': 'baichuan2-13b', 'api_type': 'baichuan2-13b', 'model_path': "./evaluation/model/baichuan2-13B", 'max_tries': 5})
        elif api_name == "fastchat-t5-3b":
            return FASTCHAT_API({"model_name": "fastchat-t5", "api_type": "fastchat", "max_tries": 5, "model_path": "./evaluation/model/fastchat-t5"})