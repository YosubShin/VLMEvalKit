import os.path as osp
import pandas as pd
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class WaltonMultimodalReasoning(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {'WaltonMultimodalColdStart': ''}
    DATASET_MD5 = {}

    def _build_vllm_judge(self, model_name, batch_size=32, **kwargs):
        """Build a VLLM-based judge model for efficient batch evaluation."""
        try:
            from vllm import LLM, SamplingParams
            import torch

            # Map model names to actual paths
            model_path = model_name
            if model_name == 'qwen3-4b':
                model_path = "Qwen/Qwen3-4B-Instruct-2507"

            # Initialize VLLM with appropriate settings
            gpu_count = torch.cuda.device_count()
            tp_size = min(gpu_count, 4)  # Use at most 4 GPUs for judge model

            vllm_params = {
                'model': model_path,
                'max_num_seqs': batch_size,
                'tensor_parallel_size': tp_size,
                'gpu_memory_utilization': 0.9,  # Make sure to unload main model before evaluation
                'max_model_len': 8192,  # Judge prompts are short
            }

            from ..smp import get_logger
            logger = get_logger('VLLM_JUDGE')

            logger.info("=== VLLM Judge Instantiation Parameters ===")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Batch size (max_num_seqs): {batch_size}")
            logger.info(f"Tensor parallel size: {tp_size}")
            logger.info(f"GPU count available: {gpu_count}")
            logger.info(f"GPU memory utilization: {vllm_params['gpu_memory_utilization']}")
            logger.info(f"Max model length: {vllm_params['max_model_len']}")
            logger.info(f"Additional kwargs: {kwargs}")
            logger.info("=" * 50)

            llm = LLM(**vllm_params)

            # Wrap in a simple interface
            class VLLMJudge:
                def __init__(self, llm_instance):
                    self.llm = llm_instance
                    self.sampling_params = SamplingParams(
                        temperature=0.1,
                        max_tokens=256,
                        stop=["```", "\n\n"]
                    )

                def generate(self, prompts):
                    """Generate responses for batch of prompts."""
                    if isinstance(prompts, str):
                        prompts = [prompts]

                    from ..smp import get_logger
                    logger = get_logger('VLLM_JUDGE')

                    logger.info("=== VLLM Judge Generation ===")
                    logger.info(f"Processing batch of {len(prompts)} prompts")
                    logger.info(f"Sampling params: temp={self.sampling_params.temperature}, max_tokens={self.sampling_params.max_tokens}")
                    logger.info("=" * 30)

                    outputs = self.llm.generate(prompts, self.sampling_params)
                    responses = [output.outputs[0].text for output in outputs]

                    logger.info(f"Generated {len(responses)} responses")
                    return responses if len(responses) > 1 else responses[0]

                def __del__(self):
                    """Clean up VLLM resources."""
                    if hasattr(self, 'llm'):
                        del self.llm
                        torch.cuda.empty_cache()

            return VLLMJudge(llm)

        except ImportError:
            # Fallback to regular judge if VLLM not available
            return build_judge(**kwargs)

    def __init__(self, dataset='WaltonMultimodalReasoning', **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset_name = dataset

    def prepare_dataset(self, dataset):
        # Load dataset from HuggingFace
        ROOT = LMUDataRoot()
        data_file = osp.join(ROOT, f'{dataset}.tsv')

        if not osp.exists(data_file):
            # Import datasets library only when needed
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "The 'datasets' library is required to load WaltonMultimodalReasoning dataset. "
                    "Please install it with: pip install datasets"
                )

            # Import encode function here to avoid circular import
            from ..tools import encode_image_to_base64

            # Load from HuggingFace
            hf_dataset = load_dataset('WaltonFuture/Multimodal-Cold-Start', split='train')

            # Convert to tsv format expected by VLMEvalKit
            data_list = []
            for idx, item in enumerate(hf_dataset):
                # The problem field contains both image reference and question
                problem_text = item['problem']

                # Handle image - if it's a PIL image, encode to base64
                image_data = ''
                if 'images' in item and item['images'] is not None:
                    # The image comes as PIL Image from HuggingFace parquet
                    try:
                        image_data = encode_image_to_base64(item['images'][0])
                    except:
                        # If it's already a string (URL or base64), use as is
                        image_data = item['images'][0]

                data_list.append({
                    'index': idx,
                    'image': image_data,
                    'question': problem_text,
                    'answer': item['answer']
                })

            # Save as TSV
            df = pd.DataFrame(data_list)
            df.to_csv(data_file, sep='\t', index=False)

        return data_file

    def load_data(self, dataset):
        data_file = self.prepare_dataset(dataset)
        return load(data_file)

    def build_prompt(self, line):
        # Build the prompt with the reasoning trace structure
        prompt = """Put your final answer within \\boxed{}.

"""

        if isinstance(line, int):
            line = self.data.iloc[line]

        # Add the question
        question = line.get('question', '')
        prompt += f"Question: {question}"

        msgs = [{'type': 'text', 'value': prompt}]

        # Add image if present
        if 'image' in line and line['image']:
            # Handle image path or base64 encoding
            image_paths = self.dump_image(line)
            # dump_image returns a list, take the first element for single image
            if image_paths:
                msgs.append({'type': 'image', 'value': image_paths[0]})

        return msgs

    def evaluate(self, eval_file, judge_model=None, **judge_kwargs):
        # Use GPT-4o-mini as judge for evaluation
        model = judge_kwargs.get('model', 'gpt-4o-mini')
        suffix = eval_file.split('.')[-1]
        result_path = eval_file.replace(f'.{suffix}', f'_{model}_judge.xlsx')
        score_path = eval_file.replace(f'.{suffix}', f'_{model}_score.csv')
        batch_size = judge_kwargs.pop('batch_size', 32)  # Use proper batch_size parameter

        if not osp.exists(result_path):
            data = load(eval_file)

            # Use provided judge model or build a new one
            if judge_model is None:
                # Check if we should use VLLM for judge (for local models like qwen3-4b)
                use_vllm_judge = judge_kwargs.get('use_vllm_judge', False)

                # Build judge model
                judge_kwargs['model'] = model
                if use_vllm_judge and not model.startswith('gpt'):
                    # Use VLLM for local judge models
                    judge_model = self._build_vllm_judge(model, batch_size=batch_size, **judge_kwargs)
                else:
                    judge_model = build_judge(**judge_kwargs)
            else:
                # Using pre-built judge model
                use_vllm_judge = hasattr(judge_model, 'llm')  # Check if it's a VLLM model

            # Check if judge is working (only for API models)
            if hasattr(judge_model, 'working'):
                assert judge_model.working(), ('WaltonMultimodalReasoning evaluation requires a working judge model\n' + DEBUG_MESSAGE)

            def extract_answer(text):
                """Extract the answer from \\boxed{} format"""
                import re
                pattern = r'\\boxed\{([^}]*)\}'
                matches = re.findall(pattern, text)
                if matches:
                    return matches[-1].strip()
                return text.strip()

            def create_judge_prompt(prediction, ground_truth):
                """Create a judge prompt for a single prediction"""
                pred_answer = extract_answer(str(prediction))
                gt_answer = extract_answer(str(ground_truth))

                return f"""You are evaluating a model's answer against the ground truth for a reasoning problem.

Model's Answer: {pred_answer}

Ground Truth: {gt_answer}

Please evaluate whether the model's answer is correct compared to the ground truth. Consider:
1. Mathematical equivalence (e.g., 58% and 58 are the same)
2. Numerical precision (allow for minor rounding differences)
3. Unit consistency (if units are provided)

Respond with a JSON containing:
{{"verdict": 1}} if the answer is correct
{{"verdict": 0}} if the answer is incorrect
"""

            def parse_judge_response(response):
                """Parse the judge's response to extract verdict"""
                try:
                    import json
                    if isinstance(response, str):
                        # Handle potential JSON extraction
                        if '```json' in response:
                            response = response.split('```json')[1].split('```')[0]
                        result = json.loads(response)
                        return result.get('verdict', 0)
                except:
                    # Fallback to 0 if parsing fails
                    return 0

            # Process in batches
            verdict_list = []
            total_items = len(data)

            from tqdm import tqdm
            with tqdm(total=total_items, desc="Evaluating with judge model") as pbar:
                for i in range(0, total_items, batch_size):
                    # Get batch of data
                    batch_end = min(i + batch_size, total_items)
                    batch_data = data.iloc[i:batch_end]

                    # Create batch of prompts
                    batch_prompts = [
                        create_judge_prompt(row['prediction'], row['answer'])
                        for _, row in batch_data.iterrows()
                    ]

                    # Generate responses for the batch
                    if use_vllm_judge:
                        # VLLM supports true batch processing
                        batch_responses = judge_model.generate(batch_prompts)
                        if not isinstance(batch_responses, list):
                            batch_responses = [batch_responses]
                    else:
                        # For HFChatModel or API models, use sequential generation
                        try:
                            if len(batch_prompts) == 1:
                                batch_responses = [judge_model.generate(batch_prompts[0])]
                            else:
                                # Try batch, but likely will be sequential
                                batch_responses = [judge_model.generate(prompt) for prompt in batch_prompts]
                        except Exception as e:
                            print(f"Error in batch generation: {e}")
                            batch_responses = [""] * len(batch_prompts)

                    # Parse responses
                    batch_verdicts = [parse_judge_response(resp) for resp in batch_responses]
                    verdict_list.extend(batch_verdicts)

                    # Update progress
                    pbar.update(batch_end - i)

            data['verdict'] = verdict_list
            dump(data, result_path)

        # Load results and compute metrics
        data = load(result_path)

        # Calculate overall accuracy
        overall_acc = data['verdict'].mean() * 100

        # Create score summary
        score_df = pd.DataFrame({
            'Metric': ['Overall Accuracy'],
            'Value': [overall_acc]
        })

        # Save score summary
        dump(score_df, score_path)

        # Return results dictionary
        ret = {'Overall': overall_acc}

        # If there are categories in the data, compute per-category accuracy
        if 'category' in data.columns:
            categories = data['category'].unique()
            for cat in categories:
                cat_data = data[data['category'] == cat]
                cat_acc = cat_data['verdict'].mean() * 100
                ret[cat] = cat_acc

        return ret