#!/usr/bin/env python3
"""
Create a TSV that trims reasoning for easy items based on k-fold evaluation.

Inputs:
- Walton TSV with columns: index, image, question, answer
- K-fold evaluated XLSX with columns: index, question, answer, prediction_1..k, verdict_1..k, verdict_sum

Rule:
- If verdict_sum >= ceil(tau * k), select the shortest correct prediction among prediction_1..k
- Else, keep original TSV answer

Output TSV columns: index, image, question, answer
"""

import argparse
import os
import os.path as osp
import logging
import math
import pandas as pd


def detect_k(evaluated_df: pd.DataFrame) -> int:
    """Detect k by counting prediction_/verdict_ columns.

    Prefer explicit verdict_ columns; fall back to prediction_ count if needed.
    """
    verdict_cols = [c for c in evaluated_df.columns if c.startswith('verdict_')]
    pred_cols = [c for c in evaluated_df.columns if c.startswith('prediction_')]
    k_v = len(verdict_cols)
    k_p = len(pred_cols)
    if k_v > 0:
        return k_v
    if k_p > 0:
        return k_p
    # Fallback: try verdict_sum max as proxy; default 16 per user context
    if 'verdict_sum' in evaluated_df.columns and pd.api.types.is_numeric_dtype(evaluated_df['verdict_sum']):
        return int(evaluated_df['verdict_sum'].max()) if evaluated_df['verdict_sum'].max() > 0 else 16
    return 16


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _coerce_index(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    if 'index' not in df.columns:
        raise ValueError(f"{source_name} missing required column 'index'")
    # Coerce to numeric then int, drop rows where index is NaN
    df['index'] = pd.to_numeric(df['index'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['index'])
    dropped = before - len(df)
    if dropped:
        logging.warning(f"{source_name}: dropped {dropped} rows with non-numeric 'index'")
    df['index'] = df['index'].astype(int)
    return df


def build_output_answers(tsv_df: pd.DataFrame, eval_df: pd.DataFrame, k_override: int = None, tau: float = 1.0) -> pd.DataFrame:
    # Ensure index alignment via explicit merge on index
    tsv_df = _normalize_columns(tsv_df)
    eval_df = _normalize_columns(eval_df)

    tsv_df = _coerce_index(tsv_df, 'TSV')
    eval_df = _coerce_index(eval_df, 'XLSX')

    # Drop potential duplicate indices in eval by keeping first occurrence
    eval_unique = eval_df.drop_duplicates(subset=['index'], keep='first')

    k_detected = detect_k(eval_unique)
    k = k_override if (k_override is not None and k_override > 0) else k_detected
    logging.info(f"Detected k={k_detected}{' (override -> ' + str(k) + ')' if k_override else ''}")
    threshold_count = int(math.ceil(float(tau) * k))
    logging.info(f"Using threshold: verdict_sum >= {threshold_count} (tau={tau}, k={k})")

    # Ensure required columns exist
    if f'prediction_1' not in eval_unique.columns:
        # Try to find a close match if columns had stray spaces
        candidates = [c for c in eval_unique.columns if c.replace(' ', '') == 'prediction_1']
        if candidates:
            eval_unique['prediction_1'] = eval_unique[candidates[0]]
        else:
            raise ValueError("Evaluated XLSX must contain 'prediction_1' column")
    # Collect all prediction_i columns up to k if available
    pred_cols = [f'prediction_{i+1}' for i in range(k) if f'prediction_{i+1}' in eval_unique.columns]
    if len(pred_cols) == 0:
        pred_cols = ['prediction_1']
    if 'verdict_sum' not in eval_unique.columns:
        # If not present, compute from verdict_i if available
        verdict_cols = [c for c in eval_unique.columns if c.startswith('verdict_')]
        if verdict_cols:
            # Coerce to numeric to handle string verdicts
            eval_unique[verdict_cols] = eval_unique[verdict_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            eval_unique['verdict_sum'] = eval_unique[verdict_cols].sum(axis=1)
        else:
            # If neither present, default to 0 to favor original answers
            eval_unique['verdict_sum'] = 0
    else:
        # Coerce verdict_sum to numeric in case it's stored as string
        eval_unique['verdict_sum'] = pd.to_numeric(eval_unique['verdict_sum'], errors='coerce').fillna(0).astype(int)

    # Collect verdict_i columns if available
    verdict_cols = [f'verdict_{i+1}' for i in range(k) if f'verdict_{i+1}' in eval_unique.columns]
    if len(verdict_cols):
        # Ensure numeric
        eval_unique[verdict_cols] = eval_unique[verdict_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Merge needed columns
    merge_cols = ['index', 'verdict_sum'] + pred_cols + verdict_cols
    merge_cols = list(dict.fromkeys([c for c in merge_cols if c in eval_unique.columns]))
    merged = pd.merge(tsv_df, eval_unique[merge_cols], on='index', how='left')

    logging.info(f"Rows in TSV: {len(tsv_df)}, rows in XLSX (unique): {len(eval_unique)}, merged rows: {len(merged)}")
    matched = merged['verdict_sum'].notna().sum()
    logging.info(f"Merged rows with evaluation: {matched}, without evaluation: {len(merged) - matched}")

    # If eval row missing, default to original answer
    def choose_answer(row):
        vs = row.get('verdict_sum')
        if pd.isna(vs):
            return row['answer']
        # Handle possible string values robustly
        try:
            vs_int = int(vs)
        except Exception:
            try:
                vs_int = int(float(str(vs).strip()))
            except Exception:
                vs_int = 0
        if vs_int < threshold_count:
            return row['answer']

        # Select among correct predictions if verdict_i columns exist
        if len(verdict_cols) and len(pred_cols) >= 1:
            # Build list of (i, pred, verd)
            candidates = []
            for i in range(k):
                pcol = f'prediction_{i+1}'
                vcol = f'verdict_{i+1}'
                if pcol in merged.columns and vcol in merged.columns:
                    pred_val = row.get(pcol)
                    verd_val = row.get(vcol)
                    try:
                        correct = int(verd_val) >= 1
                    except Exception:
                        correct = False
                    if correct and isinstance(pred_val, str) and len(pred_val) > 0:
                        candidates.append((i, pred_val))
            if candidates:
                # Choose by shortest length; tie-breaker earliest index
                candidates.sort(key=lambda t: (len(t[1]), t[0]))
                return candidates[0][1]
            # Fallback if no correct candidates despite threshold
            logging.warning("No correct prediction_i found despite threshold; falling back to prediction_1 or original answer")
            if 'prediction_1' in merged.columns and isinstance(row.get('prediction_1'), str) and len(row.get('prediction_1')):
                return row.get('prediction_1')
            return row['answer']

        # If no verdict_i columns, we cannot identify correct predictions reliably; use prediction_1 if available
        if 'prediction_1' in merged.columns and isinstance(row.get('prediction_1'), str) and len(row.get('prediction_1')):
            return row.get('prediction_1')
        return row['answer']

    merged['answer_out'] = merged.apply(choose_answer, axis=1)

    # Diagnostics
    try:
        vs_int_series = pd.to_numeric(merged['verdict_sum'], errors='coerce').fillna(-1).astype(int)
        num_selected = (vs_int_series >= threshold_count).sum()
        logging.info(f"Rows using selected prediction (mode=shortest, threshold={threshold_count}): {num_selected}")
        sample_pred_idxs = merged.loc[vs_int_series >= threshold_count, 'index'].head(5).tolist()
        if sample_pred_idxs:
            logging.info(f"Sample indices using prediction_1: {sample_pred_idxs}")
    except Exception:
        pass

    # Prepare final columns
    out = merged[['index', 'image', 'question']].copy()
    out['answer'] = merged['answer_out']
    return out


def main():
    parser = argparse.ArgumentParser(description='Trim easy Walton items using k-fold evaluation to reduce verbosity.')
    parser.add_argument('--tsv', required=True, help='Path to Walton TSV (index,image,question,answer)')
    parser.add_argument('--xlsx', required=True, help='Path to evaluated k-fold XLSX (with prediction_1..k and verdict_sum)')
    parser.add_argument('--out', required=True, help='Path to output TSV')
    parser.add_argument('--k', type=int, default=None, help='Optional override for k if detection is wrong')
    parser.add_argument('--tau', type=float, default=1.0, help='Threshold ratio for using prediction_1 (0 < tau <= 1). Default: 1.0 (all-correct)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(levelname)s: %(message)s')

    if not osp.exists(args.tsv):
        raise FileNotFoundError(f"TSV not found: {args.tsv}")
    if not osp.exists(args.xlsx):
        raise FileNotFoundError(f"XLSX not found: {args.xlsx}")

    # Load data
    tsv_df = pd.read_csv(args.tsv, sep='\t')
    xlsx_df = pd.read_excel(args.xlsx)

    # Validate required columns in TSV
    for col in ['index', 'image', 'question', 'answer']:
        if col not in tsv_df.columns:
            raise ValueError(f"Input TSV missing required column '{col}'")

    # Validate tau
    if not (args.tau is None) and not (0 < float(args.tau) <= 1.0):
        raise ValueError("--tau must be in (0, 1]")

    out_df = build_output_answers(tsv_df, xlsx_df, k_override=args.k, tau=float(args.tau))

    # Write TSV
    out_dir = osp.dirname(args.out)
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out, sep='\t', index=False)

    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == '__main__':
    main()


