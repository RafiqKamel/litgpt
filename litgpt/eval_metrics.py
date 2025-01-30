import re
import pandas as pd
import numpy as np
from amrlib.evaluate.smatch_enhanced import compute_smatch
from amrlib.evaluate.bleu_scorer import BLEUScorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import corpus_chrf
from sacrebleu import corpus_bleu
import sacrebleu
from bleurt import score
from nltk.tokenize import word_tokenize
import nltk



nltk.download("punkt")
nltk.download("wordnet")


def create_eval_dataframe(
    pred,
    sentences,
    amr_graphs,
    amr_texts,
    complexity_function,
    parse_target,
    n_sentences=None,
):
    df = pd.DataFrame()
    if parse_target == "amr":
        df["gold"] = amr_texts
    elif parse_target == "text":
        df["gold"] = sentences
    if n_sentences:
        df["n_sentences"] = n_sentences
    df["pred"] = pred
    df["sentence"] = sentences
    df["amr_graph"] = amr_graphs
    df["amrs_text"] = amr_texts
    df["complexity"] = complexity_function(df)
    return df



def bleu_scoring(preds, gold, tokenizer):
    pred_token = [tokenizer.tokenize(str(p)) for p in preds]
    gold_token = [tokenizer.tokenize(str(g)) for g in gold]
    bleu_scorer = BLEUScorer()
    bleu_score, ref_len, hyp_len = bleu_scorer.compute_bleu(refs=gold_token, hyps=pred_token)
    return bleu_score

def spring_bleu_scoring(preds, gold):
    score = corpus_bleu(preds, [gold]).score
    
    return score

def raw_corpus_bleu(hypothesis, reference,
                    offset = 0.01) -> float:
    reference = [reference]
    bleu = sacrebleu.corpus_bleu(hypothesis, reference, smooth_value=offset,
                                 force=True, use_effective_order=False,
                                 lowercase=True)
    score = bleu.score
    return score
def chrf_scoring(preds, gold):
    references = [word_tokenize(p) for p in preds]
    hypothesis = [word_tokenize(g) for g in gold]
    score = corpus_chrf(references, hypothesis)
    return score


def meteor_scoring(preds, gold):
    scores = []
    for p, g in zip(preds, gold):
        reference = word_tokenize(p)
        hypothesis = word_tokenize(g)
        score = round(meteor_score([g], p), 4)
        scores.append(score)
    return np.mean(scores)


def smatch_f1_score(preds, gold):
    preds_entries = get_entries(preds.values.tolist())
    gold_entries = get_entries(gold.values.tolist())
    score = compute_smatch(preds_entries, gold_entries)
    return score[2]


def evaluate_predictions(eval_df, evaluation_function, method):
    if method == "bottom_up":
        return eval_bottom_up(eval_df, evaluation_function)
    elif method == "top_down":
        return eval_top_down(eval_df, evaluation_function)
    elif method == "exact_level":
        return eval_at_each_level(eval_df, evaluation_function)
    else:
        raise ValueError(
            "Method not available. Avalaible methods: [bottom_up, top_down, exact_level]"
        )


def eval_bottom_up(eval_df, evaluation_function):
    max_complexity = eval_df["complexity"].max()
    min_complexity = eval_df["complexity"].min()
    previous_len = -1
    results = pd.DataFrame()
    for current_max in range(min_complexity, max_complexity + 1):
        row = {}
        df_tmp = eval_df[eval_df["complexity"] <= current_max]
        if len(df_tmp) == previous_len:
            continue
        previous_len = len(df_tmp)
        row["complexity"] = current_max
        row["n_entries"] = len(df_tmp)
        row["metrics"] = evaluation_function(
            preds=df_tmp["pred"].tolist(), gold=df_tmp["gold"].tolist()
        )
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    return results


def eval_at_each_level(eval_df, evaluation_function):
    max_complexity = eval_df["complexity"].max()
    min_complexity = eval_df["complexity"].min()
    results = pd.DataFrame()
    for current in eval_df["complexity"]:
        row = {}
        mask = eval_df['complexity'].apply(lambda x: x == current)
        df_tmp = eval_df[mask]
        if len(df_tmp) == 0:
            continue
        row["complexity"] = current
        row["n_entries"] = len(df_tmp)
        row["metrics"] = evaluation_function(
            preds=df_tmp["pred"], gold=df_tmp["gold"]
        )
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    return results


def eval_top_down(eval_df, evaluation_function):
    max_complexity = eval_df["complexity"].max()
    min_complexity = eval_df["complexity"].min()
    previous_len = -1
    results = pd.DataFrame()
    for current_min in range(max_complexity, min_complexity - 1, -1):
        row = {}
        df_tmp = eval_df[eval_df["complexity"] >= current_min]
        if len(df_tmp) == previous_len:
            continue
        previous_len = len(df_tmp)
        row["complexity"] = current_min
        row["n_entries"] = len(df_tmp)
        row["metrics"] = evaluation_function(
            preds=df_tmp["pred"], gold=df_tmp["gold"]
        )
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    return results


def get_entries(data):
    entries = []
    for e in data:
        lines = [l.strip() for l in e.splitlines()]
        lines = [l for l in lines if (l and not l.startswith("#"))]
        string = " ".join(lines)
        string = string.replace("\t", " ")  # replace tabs with a space
        # squeeze multiple spaces into a single
        string = re.sub(" +", " ", string)
        if string:
            entries.append(string)
    return entries
