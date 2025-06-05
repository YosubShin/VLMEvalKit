import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import copy as cp
import string
import pandas as pd
from .base import BaseModel
from ..smp import isimg, listinstr
from ..dataset import DATASET_TYPE


class QwenVL(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def use_custom_prompt(self, dataset):
        """Use custom prompt for specific datasets including LiveXiv."""
        if dataset in ["LiveXivTQA", "LiveXivVQA"]:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        """Build custom prompts for specific datasets."""
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        
        if dataset in ["LiveXivTQA", "LiveXivVQA"]:
            prompt = self.build_prompt_livexiv(line)
        else:
            # Default to standard prompt building
            prompt = line.get('question', '')
        
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        
        return message

    def build_prompt_livexiv(self, line, prefix=None):
        """Build prompt specifically for LiveXiv datasets."""
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        
        for key, item in options.items():
            question += f'\n{key}: {item}'
        
        if prefix is None:
            prompt = f"{question}\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = f"{prefix} {question}"

        return prompt

    def adjust_kwargs(self, dataset):
        kwargs = cp.deepcopy(self.kwargs)
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 100
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 10
        return kwargs

    def generate_inner(self, message, dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs
        prompt = ''
        for s in message:
            if s['type'] == 'image':
                prompt += f'<img>{s["value"]}</img>'
            elif s['type'] == 'text':
                prompt += s['value']
        if dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            prompt += ' Answer:'
        encoded = self.tokenizer([prompt], return_tensors='pt', padding='longest')
        input_ids = encoded.input_ids.to('cuda')
        attention_mask = encoded.attention_mask.to('cuda')

        pred = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs)
        answer = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return answer


class QwenVLChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        """Use custom prompt for specific datasets including LiveXiv."""
        if dataset in ["LiveXivTQA", "LiveXivVQA"]:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        """Build custom prompts for specific datasets."""
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        
        if dataset in ["LiveXivTQA", "LiveXivVQA"]:
            prompt = self.build_prompt_livexiv(line)
        else:
            # Default to standard prompt building
            prompt = line.get('question', '')
        
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        
        return message

    def build_prompt_livexiv(self, line, prefix=None):
        """Build prompt specifically for LiveXiv datasets."""
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        
        for key, item in options.items():
            question += f'\n{key}: {item}'
        
        if prefix is None:
            prompt = f"{question}\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = f"{prefix} {question}"

        return prompt

    def build_history(self, message):

        def concat_tilist(tilist):
            image_cnt = 1
            prompt = ''
            for item in tilist:
                if item['type'] == 'text':
                    prompt += item['value']
                elif item['type'] == 'image':
                    prompt += f"Picture {image_cnt}: <img>{item['value']}</img>\n"
                    image_cnt += 1
            return prompt

        assert len(message) % 2 == 0
        hist = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            hist.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))
        return hist

    def generate_inner(self, message, dataset=None):
        vl_list = [{'image': s['value']} if s['type'] == 'image' else {'text': s['value']} for s in message]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response

    def chat_inner(self, message, dataset=None):
        assert len(message) % 2 == 1 and message[-1]['role'] == 'user'
        history = self.build_history(message[:-1])
        vl_list = [
            {'image': s['value']} if s['type'] == 'image' else {'text': s['value']}
            for s in message[-1]['content']
        ]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=history, **self.kwargs)
        return response
