import os
import sys
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import glob
from ...smp import *
import re 
import json
FAIL_MSG = 'Failed to obtain answer via API.'

def remove_think_tags(text):
    cleaned_text = re.sub(r'\<think\>.*?\</think\>', '', text, flags=re.DOTALL)
    return cleaned_text

def post_yn_check(line):
    response = line['res']
    try:
        if '回答正确' in response:
            return True
        else:
            return False
    except:
        print(response)
        return False

def post_cmt_check(line):
    response = line['res']
    try:
        if '不一致' in response:
            return False
        elif '一致' in response and '不一致' not in response:
            return True 
        return False
    except:
        print(response)
        return False



def extract_last_bracket_content(text):
    pattern = r'\[(.*?)\]'
    matches = re.finditer(pattern, text)
    
    last_match = None
    for match in matches:
        last_match = match
    
    if last_match is None:
        return None
    return last_match.group(1)

def is_topic_right(gt, pred):
    pred = extract_last_bracket_content(pred)
    idx = 0  
    try:
        for s_1 in gt:
            if s_1 == "是" or s_1 == "否":
                while idx < len(pred) and (pred[idx] != "是" and pred[idx] != "否"):
                    idx += 1
                if idx >= len(pred) or pred[idx] != s_1:
                    return False
                idx += 1  
        while idx < len(pred):
            if pred[idx] == "是" or pred[idx] == "否":
                return False
            idx += 1
        return True
    except:
        return False


def Topic_eval(model, line):
    prompt = '''这是判断给定的视频列表中的视频是否与第一个视频属于同一个主题的结果，其中可能包含了一些分析，请帮我提取出最后的结论.内容为：”{}“。请将答案直接以列表的形式输出，列表的每一项为“是”或者“否”，表示对应视频是否与第一个视频属于同一个主题,答案形式为[是，否...是]的格式。'''
    answer = line['answer'].split("_")
    prediction = remove_think_tags(line['prediction']).strip()
    input_prompt = prompt.format(prediction)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(input_prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += res
            if is_topic_right(answer, res):
                res = "回答正确"
            else:
                res = "回答错误"
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


def build_Kwaivqa_gpt4_prompt(line):
    prompt = '''给你一个标准答案和一个回答，请根据标准答案判断回答是否正确，同时忽略回答中的分析过程中，直接对比回答中的结论和标准答案是否一致，标准答案：“{}”。回答：“{}”。请以“回答正确”或“回答错误”的格式返回答案。'''
    answer = line['answer']
    prediction = remove_think_tags(line['prediction']).strip()
    input_prompt = prompt.format(answer, prediction)
    return input_prompt

def Kwaivqa_eval(model, line):
    prompt = build_Kwaivqa_gpt4_prompt(line)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')
