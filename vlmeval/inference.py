import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.utils.batch_processing import BatchCollector, BatchProcessor, estimate_batch_processing_benefit
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.xlsx'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False,
               batch_size=None):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None:
        # Check VLLM compatibility using the model detection utility
        try:
            from vlmeval.utils.model_detection import is_model_vllm_compatible
            is_vllm_compatible = is_model_vllm_compatible(model_name)
        except ImportError:
            # Fallback to original hardcoded patterns if import fails
            is_vllm_compatible = (
                'Llama-4' in model_name
                or 'Qwen2-VL' in model_name
                or 'Qwen2.5-VL' in model_name
                or 'molmo' in model_name.lower()
            )

        if is_vllm_compatible:
            kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    # Check if we can use batch processing
    supports_batching = (
        use_vllm and
        hasattr(model, 'supports_batch_processing') and
        model.supports_batch_processing() and
        batch_size is not None and
        batch_size > 1
    )

    if supports_batching:
        # Use batch processing for VLLM models
        if verbose:
            benefit = estimate_batch_processing_benefit(lt, batch_size)
            print(f"Using VLLM batch processing: estimated {benefit['speedup']}x speedup "
                  f"({benefit['time_saved_percent']}% time saved)")

        res.update(infer_data_batch(model, dataset, data, dataset_name, batch_size, verbose, res, out_file))
    else:
        # Use sequential processing (original behavior)
        if batch_size is not None and batch_size > 1 and verbose:
            print(f"Batch processing not available for {model_name}, using sequential processing")

        for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
            idx = data.iloc[i]['index']
            if idx in res:
                continue

            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(data.iloc[i])

            response = model.generate(message=struct, dataset=dataset_name)
            torch.cuda.empty_cache()

            if verbose:
                print(response, flush=True)

            res[idx] = response
            if (i + 1) % 10 == 0:
                dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


def infer_data_batch(model, dataset, data, dataset_name, batch_size, verbose=False, existing_res=None, out_file=None):
    """Perform batch inference using VLLM models."""
    results = {}
    existing_res = existing_res or {}
    processed_count = 0

    # Initialize batch collector and processor
    collector = BatchCollector(
        max_batch_size=batch_size,
        batch_timeout=5.0,  # 5 seconds timeout
        enable_smart_batching=True,
        verbose=verbose
    )

    processor = BatchProcessor(model, verbose=verbose)

    # Count total items and items that need processing
    total_items = len(data)
    items_to_process = 0
    for _, row in data.iterrows():
        idx = row['index']
        if idx not in existing_res:
            items_to_process += 1

    model_desc = model.model_name if hasattr(model, "model_name") else "Model"
    progress_bar = tqdm(total=total_items, desc=f'Batch Infer {model_desc}/{dataset_name}')

    # Update progress bar for already processed items
    already_processed = total_items - items_to_process
    if already_processed > 0:
        progress_bar.update(already_processed)
        if verbose:
            print(f"Reusing {already_processed} existing results from previous run")

    # Add all items to collector (skip already processed ones)
    for i, row in data.iterrows():
        idx = row['index']

        # Skip if already processed (reuse logic)
        if idx in existing_res:
            results[idx] = existing_res[idx]
            continue

        # Build prompt
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(row, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(row)

        # Try to add to batch
        ready_batch = collector.add_item(idx, struct, dataset_name)

        if ready_batch:
            # Process the ready batch
            batch_results = processor.process_batch(ready_batch)
            for item_idx, response in batch_results:
                results[item_idx] = response
                progress_bar.update(1)
                processed_count += 1

                if verbose:
                    print(f"Batch result for {item_idx}: {response[:50]}...", flush=True)

                # Periodic save (every 10 items, like sequential processing)
                if out_file and processed_count % 10 == 0:
                    # Combine existing results with new results for saving
                    combined_results = {**existing_res, **results}
                    dump(combined_results, out_file)

    # Process any remaining items
    remaining_batches = collector.flush_all()
    for batch in remaining_batches:
        batch_results = processor.process_batch(batch)
        for item_idx, response in batch_results:
            results[item_idx] = response
            progress_bar.update(1)
            processed_count += 1

            if verbose:
                print(f"Final batch result for {item_idx}: {response[:50]}...", flush=True)

            # Periodic save (every 10 items, like sequential processing)
            if out_file and processed_count % 10 == 0:
                # Combine existing results with new results for saving
                combined_results = {**existing_res, **results}
                dump(combined_results, out_file)

    progress_bar.close()

    # Print statistics
    if verbose:
        stats = collector.get_stats()
        print("Batch processing completed:")
        print(f"  Total items: {stats['total_collected']}")
        print(f"  Total batches: {stats['total_batches_sent']}")
        print(f"  Average batch size: {stats['avg_batch_size']:.1f}")

    # Clear GPU cache after batch processing
    torch.cuda.empty_cache()

    return results


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False,
    batch_size=None
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm, batch_size=batch_size)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
