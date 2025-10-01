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
            hf_dataset = load_dataset('oumi-ai/walton-multimodal-cold-start-r1-format', split='train')

            # Convert to tsv format expected by VLMEvalKit
            data_list = []
            for idx, item in enumerate(hf_dataset):
                # The problem field contains both image reference and question
                problem_text = item['problem']

                # Handle image - if it's a PIL image, encode to base64
                image_data = ''
                if 'image' in item and item['image'] is not None:
                    # The image comes as PIL Image from HuggingFace parquet
                    try:
                        image_data = encode_image_to_base64(item['image'])
                    except:
                        # If it's already a string (URL or base64), use as is
                        image_data = item['image']

                data_list.append({
                    'index': idx,
                    'image': image_data,
                    'question': problem_text,
                    'answer': item['solution']
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
        prompt = """When analyzing any query or task, please follow the structure below:
1. Draft Response:
Generate an initial response.
2. Critical Comments:
Analyze your draft response by considering:
• Potential weaknesses or gaps
• Logical flaws or inconsistencies
• Missing perspectives or alternatives
• Areas for improvement
• Suggestions for a better version
• Steering toward the given answer
The critical comments should:
• Be specific and actionable
• Reference particular parts of the draft
• Suggest concrete improvements
• Consider different angles or approaches
• Guide towards a more comprehensive solution
Output Format:
• Draft Response:
Your initial complete response to the instruction.
• Critical Comments:
Your analysis of the draft response, highlighting areas for improvement and suggesting specific enhancements.
• Final Answer:
Put your final answer within \\boxed{}.

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

    def evaluate(self, eval_file, **judge_kwargs):
        # Use GPT-4o-mini as judge for evaluation
        model = judge_kwargs.get('model', 'gpt-4o-mini')
        suffix = eval_file.split('.')[-1]
        result_path = eval_file.replace(f'.{suffix}', f'_{model}_judge.xlsx')
        score_path = eval_file.replace(f'.{suffix}', f'_{model}_score.csv')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(result_path):
            data = load(eval_file)

            # Build judge model
            judge_kwargs['model'] = model
            judge_model = build_judge(**judge_kwargs)

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

            def judge_one(model, line):
                """Judge a single prediction against ground truth"""
                prediction = extract_answer(str(line.get('prediction', '')))
                ground_truth = extract_answer(str(line.get('answer', '')))

                # Build judge prompt
                judge_prompt = f"""You are evaluating a model's answer against the ground truth for a reasoning problem.

Model's Answer: {prediction}

Ground Truth: {ground_truth}

Please evaluate whether the model's answer is correct compared to the ground truth. Consider:
1. Mathematical equivalence (e.g., 58% and 58 are the same)
2. Numerical precision (allow for minor rounding differences)
3. Unit consistency (if units are provided)

Respond with a JSON containing:
{{"verdict": 1}} if the answer is correct
{{"verdict": 0}} if the answer is incorrect
"""

                response = judge_model.generate(judge_prompt)

                # Parse response
                try:
                    import json
                    if isinstance(response, str):
                        # Handle potential JSON extraction
                        if '```json' in response:
                            response = response.split('```json')[1].split('```')[0]
                        result = json.loads(response)
                        return result.get('verdict', 0)
                except:
                    # Fallback to simple string comparison if JSON parsing fails
                    return 1 if prediction.lower() == ground_truth.lower() else 0

            # Judge all predictions
            verdict_list = track_progress_rich(
                lambda line: judge_one(judge_model, line),
                [data.iloc[i] for i in range(len(data))],
                nproc=nproc,
                chunksize=nproc,
            )

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