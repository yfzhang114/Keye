from .image_mcq import ImageMCQDataset
from .image_base import ImageBaseDataset 
from .image_vqa import ImageVQADataset 
from .image_yorn import ImageYORNDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import *
import pandas as pd
import warnings
import re

def get_options(text):
    pattern = r'([A-Z])\.\s*(.+)'
    matches = re.findall(pattern, text)
    options = {key: value.strip() for key, value in matches}

    return options
def remove_think_tags(text):
    cleaned_text = re.sub(r'\<think\>.*?\</think\>', '', text, flags=re.DOTALL)
    return cleaned_text

class KwaiYORNDataset(ImageYORNDataset):
    DATASET_URL = {
        'PornComment': 'PornComment.tsv',
        'High_like': 'High_like.tsv',
        'SPU': 'SPU.tsv'
    }
    
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text' :
                item['value'] += '\n请直接给出结论，回答是或者否，不需要做额外分析，并且不许回答不确定'
        return msgs
    

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.yorn import YOrN_Extraction, YOrN_cn_Extraction, YOrN_auxeval,YOrN_cn_auxeval
        from .utils.yorn import default_rating

        dataset = self.dataset_name
        data = load(eval_file)
        data['prediction'] = [remove_think_tags(str(x)) for x in data['prediction']]
        storage = eval_file.replace('.xlsx', '_auxmatch.xlsx')
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            ans_map = {k: YOrN_cn_Extraction(v) for k, v in zip(data['index'], data['prediction'])}
            if osp.exists(tmp_file):
                tmp = load(tmp_file)
                for k in tmp:
                    if ans_map[k] == 'Unknown' and tmp[k] != 'Unknown':
                        ans_map[k] = tmp[k]

            data['extracted'] = [ans_map[x] for x in data['index']]
            unknown = data[data['extracted'] == 'Unknown']

            model = judge_kwargs.get('model', 'exact_matching')
            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                model = None
                warnings.warn('OPENAI_API_KEY is not working properly, will use exact matching for evaluation')
            
            if model is not None:
                lt = len(unknown)
                lines = [unknown.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = list(unknown['index'])
                if len(tups):
                    res = track_progress_rich(
                        YOrN_cn_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            data['extracted'] = [ans_map[x] for x in data['index']]
            dump(data, storage)

        data = load(storage)
        data['score'] = (data['answer'] == data['extracted'])
        dump(data, storage)
        score = default_rating(storage)
        score_tgt = eval_file.replace('.xlsx', '_score.csv')
        dump(score, score_tgt)
        return score




class KwaiVQADataset(ImageVQADataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'Video_Order': 'Video_Order.tsv',
        'Video_Topic': 'Video_Topic.tsv',
        'CPV': 'CPV.tsv'
    }

    def judge_acc(self, result_file, post_check):
        def default_value():
            return [0, 0]
        data = load(result_file)
        score_count = defaultdict(default_value)
        for i in range(len(data)):
            item = data.iloc[i]
            output_flag = post_check(item)
            score_count['Overall'][0] += output_flag
            score_count['Overall'][1] += 1
        final_acc = dict()
        final_acc['Overall'] = [score_count['Overall'][0] / score_count['Overall'][1]]
        ret = pd.DataFrame(final_acc)
        return ret


    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.kwaivqa import post_yn_check, Kwaivqa_eval, Topic_eval

        model = judge_kwargs.get('model', 'gpt-4o-mini')
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            
            model = build_judge(max_tokens=128, **judge_kwargs)

            if model is not None:
                lt = len(data)
                lines = [data.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = [line['index'] for line in lines]
            
                if len(indices):
                    if 'Video_Topic' in self.dataset_name:
                        eval_process = Topic_eval
                    else:
                        eval_process = Kwaivqa_eval
                    new_results = track_progress_rich(
                        eval_process,
                        tups,
                        nproc=nproc,
                        chunksize=nproc,
                        keys=indices,
                        save=tmp_file,
                    )
                    ans = load(tmp_file)
                
                data['res'] = [ans[idx]['res'] for idx in data['index']]
                data['log'] = [ans[idx]['log'] for idx in data['index']]
                dump(data, storage)

                post_check = post_yn_check
                score = self.judge_acc(storage, post_check)
                score_pth = storage.replace('.xlsx', '_acc.csv')
                dump(score, score_pth)
        return score
        


