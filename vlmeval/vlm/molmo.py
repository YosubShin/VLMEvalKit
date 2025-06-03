import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from mimetypes import guess_type
from io import BytesIO
import base64
import os
import logging

TYPE_PROMPTS = {
    'Y/N':'vqa2:',
    'VQA':'vqa2:',
    'MCQ':'a_okvqa_mc:',
}

DATASET_PROMPTS = {
    'AI2D_TEST':'ai2_diagram:',
    'AI2D_TEST_NO_MASK':'ai2_diagram:',
    'COCO_VAL':'coco_captioning:',
    'ChartQA_TEST':'chart_qa:',
    'ChartQA_VAL':'chart_qa:',
    'DocVQA_VAL':'doc_qa:',
    'DocVQA_TEST':'doc_qa:',
    'InfoVQA_TEST':'info_qa:',
    'InfoVQA_VAL':'info_qa:',
    'OCRVQA_TEST':'ocr_vqa:',
    'OCRVQA_TESTCORE':'ocr_vqa:',
    'ScienceQA_VAL':'science_qa:',
    'ScienceQA_TEST':'science_qa:',
    'TableVQABench':'tabwmp_da:',
    'TextVQA_VAL':'text_vqa:'
}

VLLM_MAX_IMAGE_INPUT_NUM = 24
DEFAULT_MAX_CONTEXT_LENGTH = 4096


class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='allenai/Molmo-7B-D-0924', **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            import einops
        except Exception as e:
            logging.critical('Please install transformer and einops before using molmo.')
            raise e

        self.model_path = model_path
        self.kwargs = kwargs
        self.model_name = model_path
        # set default maximum number of crops to 36
        self.max_crops = kwargs.get('max_crops', 36)
        
        # VLLM configuration
        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        
        # Generation parameters
        self.max_new_tokens = kwargs.get('max_new_tokens', 200)
        self.temperature = kwargs.get('temperature', 0.0)
        self.verbose = kwargs.get('verbose', False)
        
        # Context length management
        self.max_context_length = kwargs.get('max_context_length', DEFAULT_MAX_CONTEXT_LENGTH)
        self.auto_truncate = kwargs.get('auto_truncate', True)
        
        if self.use_vllm:
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. '
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
                # Automatically set it for this process
                os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            
            # Determine appropriate max_model_len for Molmo
            # Molmo models typically have max_position_embeddings=4096
            max_model_len = kwargs.get("max_model_len", None)
            if max_model_len is None:
                # Use model's actual context length (4096 for most Molmo models)
                max_model_len = self.max_context_length
                
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=4,
                max_model_len=max_model_len,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,  # Required for Molmo
            )
            
        else:
            if '72b' not in model_path.lower():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map='cuda')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map='auto')

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA']:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        prefix = None
        if dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            prompt = self.build_prompt_mcq_vqa(line)
        elif dataset in ['MathVista_MINI']:
            prompt = self.build_prompt_mathvista(line)
        elif dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            prompt = self.build_prompt_ai2d(line)
        elif dataset is not None and listinstr(list(DATASET_PROMPTS.keys()), dataset):
            prefix = DATASET_PROMPTS[dataset]  # rest of supervised datasets are in VQA format
            prompt = self.build_prompt_vqa(line, prefix)
        elif dataset is not None and listinstr(['MCQ'], DATASET_TYPE(dataset)):
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)
        return message

    def build_prompt_mathvista(self, line):
        if line['question_type'] == 'multi_choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_ai2d(self, line):
        def option_is_abc(line):
            for cand in string.ascii_uppercase:
                if cand in line and not pd.isna(line[cand]):
                    # check if option is single letter
                    if not line[cand].strip().isalpha() or len(line[cand].strip()) > 1:
                        return False
            return True

        if line['abcLabel'] and option_is_abc(line):
            prompt = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                prompt += f'\n{item}'
            prompt = f"ai2_diagram_no_letter: {prompt}"
            # prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram_no_letter:')
        else:
            prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram:')
        return prompt

    def build_prompt_mcq_vqa(self, line):
        if line['question_type'] == 'multiple-choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_multiple_choice(self, line, prefix=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}: {item}'
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['MCQ']} {question}"
        else:
            prompt = f"{prefix} {question}"

        return prompt

    def build_prompt_vqa(self, line, prefix=None):
        question = line['question']
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['VQA']} {question}"
        else:
            prompt = f"{prefix} {question}"
        return prompt

    def _prepare_content_vllm(self, inputs: list, dataset: str = None) -> list:
        """Prepare content for VLLM inference."""
        content = []
        num_images = 0
        
        for item in inputs:
            if item['type'] == 'text':
                content.append({
                    "type": "text",
                    "text": item['value']
                })
            elif item['type'] == 'image':
                if num_images < self.limit_mm_per_prompt:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": self._ensure_image_url(item['value'])
                        }
                    })
                    num_images += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
        
        # Apply automatic truncation if enabled
        if self.auto_truncate:
            content = self._truncate_content(content, self.max_context_length)
        
        return content
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _truncate_content(self, content: list, max_tokens: int) -> list:
        """Truncate content to fit within max_tokens while preserving images and structure."""
        if not self.auto_truncate:
            return content
            
        # Separate images and text
        text_items = [item for item in content if item.get('type') == 'text']
        image_items = [item for item in content if item.get('type') in ['image', 'image_url']]
        
        # Estimate tokens for images (conservative estimate: ~100 tokens per image)
        image_tokens = len(image_items) * 100
        
        # Calculate available tokens for text
        available_text_tokens = max_tokens - image_tokens - self.max_new_tokens - 100  # Buffer
        
        if available_text_tokens <= 0:
            if self.verbose:
                logging.warning(f"Too many images ({len(image_items)}) for context length. Keeping first {max_tokens // 200} images.")
            # Keep only essential images
            max_images = max(1, (max_tokens - self.max_new_tokens - 100) // 200)
            image_items = image_items[:max_images]
            available_text_tokens = 100  # Minimal text
        
        # Combine all text content
        all_text = " ".join([item.get('text', item.get('value', '')) for item in text_items])
        
        # Estimate current text tokens
        text_tokens = self._estimate_token_count(all_text)
        
        if text_tokens > available_text_tokens:
            if self.verbose:
                logging.warning(
                    f"Text content ({text_tokens} tokens) exceeds available space ({available_text_tokens} tokens). "
                    f"Truncating to fit context length of {self.max_context_length}."
                )
            
            # Calculate truncation ratio
            truncation_ratio = available_text_tokens / text_tokens
            target_length = int(len(all_text) * truncation_ratio)
            
            # Prefer keeping the beginning and end of text (remove middle)
            if target_length > 0:
                start_length = target_length // 2
                end_length = target_length - start_length
                
                if len(all_text) > start_length + end_length + 50:  # Leave space for truncation indicator
                    truncated_text = (
                        all_text[:start_length] + 
                        " ... [TRUNCATED] ... " + 
                        all_text[-end_length:] if end_length > 0 else ""
                    )
                else:
                    truncated_text = all_text[:target_length]
            else:
                truncated_text = all_text[:200]  # Minimal fallback
            
            # Update text items
            if text_items:
                text_items[0]['text'] = truncated_text
                text_items[0]['value'] = truncated_text
                text_items = text_items[:1]  # Keep only first text item
        
        # Reconstruct content maintaining order
        truncated_content = []
        text_idx = 0
        image_idx = 0
        
        for item in content:
            if item.get('type') == 'text' and text_idx < len(text_items):
                truncated_content.append(text_items[text_idx])
                text_idx += 1
            elif item.get('type') in ['image', 'image_url'] and image_idx < len(image_items):
                truncated_content.append(image_items[image_idx])
                image_idx += 1
                
        return truncated_content

    def _ensure_image_url(self, image_path: str) -> str:
        """Convert image path to URL format for VLLM."""
        prefixes = ['http://', 'https://', 'file://', 'data:image']
        if any(image_path.startswith(prefix) for prefix in prefixes):
            return image_path
        if os.path.exists(image_path):
            return 'file://' + os.path.abspath(image_path)
        raise ValueError(f'Invalid image path: {image_path}')
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for VLLM."""
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        
        image = Image.open(image_path)
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)
        
        with BytesIO() as output:
            image.convert("RGB").save(output, format="JPEG")
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        
        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    @staticmethod
    def _rgba_to_rgb(image):
        """Convert RGBA image to RGB."""
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")
    
    def generate_inner_vllm(self, message, dataset=None):
        """Generate response using VLLM."""
        from vllm import SamplingParams
        
        # Convert message to VLLM format
        content = self._prepare_content_vllm(message, dataset=dataset)
        
        # Handle multimodal inputs
        images = []
        text_parts = []
        
        for item in content:
            if item["type"] == "text":
                text_parts.append(item["text"])
            elif item["type"] == "image_url":
                image_url = item["image_url"]["url"]
                if image_url.startswith('file://'):
                    image_path = image_url[7:]  # Remove 'file://' prefix
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    images.append(image)
        
        # Combine text parts
        prompt = " ".join(text_parts)
        
        if self.verbose:
            print(f'\\033[31mVLLM Prompt: {prompt}\\033[0m')
            print(f'\\033[31mVLLM Images: {len(images)}\\033[0m')
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=["<|endoftext|>"]  # Molmo stop token
        )
        
        # Generate with VLLM
        if images:
            outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images},
                },
                sampling_params=sampling_params,
            )
        else:
            outputs = self.llm.generate(
                {"prompt": prompt},
                sampling_params=sampling_params,
            )
        
        # Extract generated text
        generated_text = ""
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
        
        # Apply post-processing specific to Molmo/dataset
        if dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            if 'ai2_diagram_no_letter' in prompt:
                try:
                    options = prompt.split('\\n')[1:]
                    if generated_text in options:
                        answer = options.index(generated_text)
                        generated_text = chr(answer + ord('A'))
                except (IndexError, ValueError):
                    pass  # Keep original text if parsing fails
        
        if self.verbose:
            print(f'\\033[32mVLLM Generated: {generated_text}\\033[0m')
        
        return generated_text

    def generate_inner_transformers(self, message, dataset=None):
        from transformers import GenerationConfig
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        # Apply truncation to prompt if needed
        if self.auto_truncate:
            estimated_tokens = self._estimate_token_count(prompt)
            available_tokens = self.max_context_length - self.max_new_tokens - 200  # Buffer for image tokens
            
            if estimated_tokens > available_tokens:
                if self.verbose:
                    logging.warning(
                        f"Prompt too long ({estimated_tokens} tokens). Truncating to {available_tokens} tokens."
                    )
                # Calculate target length and truncate prompt
                truncation_ratio = available_tokens / estimated_tokens
                target_length = int(len(prompt) * truncation_ratio)
                
                # Keep beginning and end
                start_length = target_length // 2
                end_length = target_length - start_length
                
                if len(prompt) > start_length + end_length + 50:
                    prompt = (
                        prompt[:start_length] + 
                        " ... [TRUNCATED] ... " + 
                        prompt[-end_length:] if end_length > 0 else ""
                    )
                else:
                    prompt = prompt[:target_length]

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # process the image and text
        max_crops = self.max_crops
        inputs = self.processor.process(
            images=[image],
            text=prompt,
            images_kwargs={
                "max_crops": max_crops
            }
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # AI2D: map direct answer to letter option
        if dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            # 'ai2_diagram_no_letter: Which of the following is the magma chamber?\nK\nB\nC\nH'
            if 'ai2_diagram_no_letter' in prompt:
                options = prompt.split('\n')[1:]
                answer = options.index(generated_text)
                generated_text = chr(answer + ord('A'))

        # print(dataset, prompt, generated_text, inputs['images'].size()) # uncomment to debug

        return generated_text
    
    def generate_inner(self, message, dataset=None):
        """Route to appropriate generation method."""
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
