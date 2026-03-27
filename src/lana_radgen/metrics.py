import os
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from transformers import BertTokenizer

try:
    from radgraph import F1RadGraph
except Exception as radgraph_import_error:  # pragma: no cover - optional dependency path
    F1RadGraph = None
    _RADGRAPH_IMPORT_ERROR = radgraph_import_error
else:  # pragma: no cover - optional dependency path
    _RADGRAPH_IMPORT_ERROR = None


if not hasattr(BertTokenizer, "encode_plus"):  # pragma: no cover - compatibility shim for radgraph on Transformers 5.x
    def _compat_encode_plus(self, *args, **kwargs):
        return self._encode_plus(*args, **kwargs)

    BertTokenizer.encode_plus = _compat_encode_plus

if not hasattr(BertTokenizer, "build_inputs_with_special_tokens"):  # pragma: no cover - compatibility shim for radgraph on Transformers 5.x
    def _compat_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls_token_id = self.cls_token_id if self.cls_token_id is not None else self.convert_tokens_to_ids("[CLS]")
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else self.convert_tokens_to_ids("[SEP]")
        if token_ids_1 is None:
            return [cls_token_id] + list(token_ids_0) + [sep_token_id]
        return [cls_token_id] + list(token_ids_0) + [sep_token_id] + list(token_ids_1) + [sep_token_id]

    BertTokenizer.build_inputs_with_special_tokens = _compat_build_inputs_with_special_tokens


CHEXPERT_14_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

CHEXPERT_5_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CHEXPERT_LABEL_PATTERNS = {
    "Enlarged Cardiomediastinum": [
        r"\benlarged cardiomediastinum\b",
        r"\bcardiomediastinal silhouette is enlarged\b",
        r"\bcardiomediastinal enlargement\b",
        r"\bmediastinal widening\b",
    ],
    "Cardiomegaly": [r"\bcardiomegaly\b", r"\benlarged cardiac silhouette\b", r"\bheart size is enlarged\b"],
    "Lung Opacity": [
        r"\blung opacity\b",
        r"\bairspace opacity\b",
        r"\bairspace opacities\b",
        r"\bopacity in the (?:right|left|upper|lower|mid)\b",
        r"\bopacification\b",
        r"\bpatchy opacity\b",
        r"\bpatchy opacities\b",
        r"\bfocal opacity\b",
    ],
    "Lung Lesion": [r"\blung lesion\b", r"\bpulmonary nodule\b", r"\bpulmonary mass\b", r"\blung mass\b", r"\blung nodule\b"],
    "Edema": [r"\bedema\b", r"\bpulmonary edema\b", r"\bvascular congestion\b", r"\binterstitial edema\b"],
    "Consolidation": [r"\bconsolidation\b", r"\bconsolidative\b"],
    "Pneumonia": [r"\bpneumonia\b", r"\binfectious infiltrate\b"],
    "Atelectasis": [r"\batelectasis\b", r"\bsubsegmental atelectatic\b", r"\bplate-like atelectatic\b"],
    "Pneumothorax": [r"\bpneumothorax\b"],
    "Pleural Effusion": [r"\bpleural effusion\b", r"\bpleural effusions\b", r"\beffusion\b"],
    "Pleural Other": [r"\bpleural thickening\b", r"\bpleural plaques?\b", r"\bpleural abnormality\b"],
    "Fracture": [r"\bfracture\b", r"\bfractures\b", r"\bfractured\b"],
    "Support Devices": [
        r"\bsupport devices?\b",
        r"\bpacemaker\b",
        r"\bpicc\b",
        r"\bport-a-cath\b",
        r"\bportacath\b",
        r"\bendotracheal tube\b",
        r"\bet tube\b",
        r"\benteric tube\b",
        r"\bng tube\b",
        r"\bchest tube\b",
        r"\bcentral venous catheter\b",
        r"\bcatheter tip\b",
        r"\bline tip\b",
        r"\bpostoperative changes\b",
    ],
    "No Finding": [r"\bno acute cardiopulmonary abnormality\b", r"\bno acute disease\b", r"\bno focal airspace disease\b", r"\bno finding\b"],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text))


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def corpus_bleu_4(predictions: Sequence[str], references: Sequence[str], max_n: int = 4) -> float:
    clipped_totals = [0.0] * max_n
    pred_totals = [0.0] * max_n
    pred_len = 0
    ref_len = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)
        for n in range(1, max_n + 1):
            pred_ng = _ngrams(pred_tokens, n)
            ref_ng = _ngrams(ref_tokens, n)
            pred_totals[n - 1] += sum(pred_ng.values())
            clipped_totals[n - 1] += sum(min(count, ref_ng[ng]) for ng, count in pred_ng.items())

    precisions = []
    for clipped, total in zip(clipped_totals, pred_totals):
        precisions.append((clipped + 1.0) / (total + 1.0))

    if pred_len == 0:
        return 0.0
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / max(pred_len, 1)))
    return float(brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n))


def corpus_bleu_1(predictions: Sequence[str], references: Sequence[str]) -> float:
    return corpus_bleu_4(predictions, references, max_n=1)


def _lcs_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    dp = [[0] * (len(tokens_b) + 1) for _ in range(len(tokens_a) + 1)]
    for idx_a, token_a in enumerate(tokens_a, start=1):
        for idx_b, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                dp[idx_a][idx_b] = dp[idx_a - 1][idx_b - 1] + 1
            else:
                dp[idx_a][idx_b] = max(dp[idx_a - 1][idx_b], dp[idx_a][idx_b - 1])
    return dp[-1][-1]


def rouge_l(predictions: Sequence[str], references: Sequence[str], beta: float = 1.2) -> float:
    scores = []
    beta_sq = beta * beta
    for prediction, reference in zip(predictions, references):
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            scores.append(0.0)
            continue
        precision = lcs / max(len(pred_tokens), 1)
        recall = lcs / max(len(ref_tokens), 1)
        denom = recall + beta_sq * precision
        if denom == 0:
            scores.append(0.0)
        else:
            scores.append(((1 + beta_sq) * precision * recall) / denom)
    return float(sum(scores) / max(len(scores), 1))


def meteor_score(predictions: Sequence[str], references: Sequence[str], alpha: float = 0.9, gamma: float = 0.5, beta: float = 3.0) -> float:
    scores = []
    for prediction, reference in zip(predictions, references):
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        if not pred_tokens and not ref_tokens:
            scores.append(1.0)
            continue
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        ref_counter = Counter(ref_tokens)
        matched_flags = []
        matched_positions = []
        matches = 0
        for idx, token in enumerate(pred_tokens):
            if ref_counter[token] > 0:
                ref_counter[token] -= 1
                matches += 1
                matched_flags.append(True)
                try:
                    matched_positions.append(ref_tokens.index(token, matched_positions[-1] + 1 if matched_positions else 0))
                except ValueError:
                    matched_positions.append(ref_tokens.index(token))
            else:
                matched_flags.append(False)

        if matches == 0:
            scores.append(0.0)
            continue

        precision = matches / len(pred_tokens)
        recall = matches / len(ref_tokens)
        denom = alpha * precision + (1 - alpha) * recall
        f_mean = 0.0 if denom == 0 else (precision * recall) / denom

        chunks = 0
        previous_matched = False
        for matched in matched_flags:
            if matched and not previous_matched:
                chunks += 1
            previous_matched = matched
        penalty = gamma * ((chunks / matches) ** beta)
        scores.append((1 - penalty) * f_mean)

    return float(sum(scores) / max(len(scores), 1))


def cider_d(predictions: Sequence[str], references: Sequence[str], max_n: int = 4) -> float:
    refs_by_n = [Counter() for _ in range(max_n)]
    document_count = max(len(references), 1)
    for reference in references:
        ref_tokens = _tokenize(reference)
        for n in range(1, max_n + 1):
            refs_by_n[n - 1].update(_ngrams(ref_tokens, n).keys())

    def tfidf_vector(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], float]:
        ngram_counts = _ngrams(tokens, n)
        total = sum(ngram_counts.values()) or 1
        vector = {}
        for ngram, count in ngram_counts.items():
            df = refs_by_n[n - 1].get(ngram, 0)
            idf = math.log((document_count + 1.0) / (df + 1.0))
            vector[ngram] = (count / total) * idf
        return vector

    def cosine_similarity(vec_a: Dict[Tuple[str, ...], float], vec_b: Dict[Tuple[str, ...], float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    scores = []
    for prediction, reference in zip(predictions, references):
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        sim_sum = 0.0
        for n in range(1, max_n + 1):
            pred_vec = tfidf_vector(pred_tokens, n)
            ref_vec = tfidf_vector(ref_tokens, n)
            sim_sum += cosine_similarity(pred_vec, ref_vec)
        scores.append(10.0 * sim_sum / max_n)
    return float(sum(scores) / max(len(scores), 1))


def _extract_chexpert_labels(report: str) -> set[str]:
    normalized = _normalize_text(report)
    labels = set()
    for label, patterns in CHEXPERT_LABEL_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            labels.add(label)
    return labels


def _compute_chexpert_f1(predicted_labels_per_report: Sequence[set[str]], reference_labels: Sequence[Dict[str, int]], label_names: Sequence[str]) -> Dict[str, object]:
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for pred_labels, ref_label_map in zip(predicted_labels_per_report, reference_labels):
        for label in label_names:
            in_pred = label in pred_labels
            in_ref = bool(ref_label_map.get(label, 0))
            if in_pred and in_ref:
                tp[label] += 1
            elif in_pred and not in_ref:
                fp[label] += 1
            elif in_ref and not in_pred:
                fn[label] += 1

    per_label_f1 = {}
    for label in label_names:
        precision = tp[label] / max(tp[label] + fp[label], 1)
        recall = tp[label] / max(tp[label] + fn[label], 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_label_f1[label] = float(f1)

    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())
    micro_precision = tp_total / max(tp_total + fp_total, 1)
    micro_recall = tp_total / max(tp_total + fn_total, 1)
    micro_f1 = 0.0 if micro_precision + micro_recall == 0 else float(2 * micro_precision * micro_recall / (micro_precision + micro_recall))
    macro_f1 = float(sum(per_label_f1.values()) / max(len(per_label_f1), 1))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "chexpert_per_label_f1": per_label_f1,
        "chexpert_label_names": list(label_names),
    }


def _reference_label_maps_from_reports(references: Sequence[str]) -> List[Dict[str, int]]:
    reference_label_maps = []
    for reference in references:
        ref_labels = _extract_chexpert_labels(reference)
        reference_label_maps.append({label: int(label in ref_labels) for label in CHEXPERT_14_LABELS})
    return reference_label_maps


def chexpert_label_f1(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, object]:
    predicted_labels_per_report = [_extract_chexpert_labels(prediction) for prediction in predictions]
    reference_label_maps = _reference_label_maps_from_reports(references)
    chexpert_14 = _compute_chexpert_f1(predicted_labels_per_report, reference_label_maps, CHEXPERT_14_LABELS)
    chexpert_5 = _compute_chexpert_f1(predicted_labels_per_report, reference_label_maps, CHEXPERT_5_LABELS)
    return {
        "chexpert_f1_14_micro": chexpert_14["micro_f1"],
        "chexpert_f1_14_macro": chexpert_14["macro_f1"],
        "chexpert_f1_5_micro": chexpert_5["micro_f1"],
        "chexpert_f1_5_macro": chexpert_5["macro_f1"],
        "chexpert_f1_micro": chexpert_14["micro_f1"],
        "chexpert_f1_macro": chexpert_14["macro_f1"],
        "chexpert_per_label_f1": chexpert_14["chexpert_per_label_f1"],
        "chexpert_label_names": chexpert_14["chexpert_label_names"],
        "chexpert_5_label_names": list(CHEXPERT_5_LABELS),
    }


def chexpert_label_f1_from_reference_labels(predictions: Sequence[str], reference_labels: Sequence[Dict[str, int]]) -> Dict[str, object]:
    predicted_labels_per_report = [_extract_chexpert_labels(prediction) for prediction in predictions]
    chexpert_14 = _compute_chexpert_f1(predicted_labels_per_report, reference_labels, CHEXPERT_14_LABELS)
    chexpert_5 = _compute_chexpert_f1(predicted_labels_per_report, reference_labels, CHEXPERT_5_LABELS)
    return {
        "chexpert_f1_14_micro": chexpert_14["micro_f1"],
        "chexpert_f1_14_macro": chexpert_14["macro_f1"],
        "chexpert_f1_5_micro": chexpert_5["micro_f1"],
        "chexpert_f1_5_macro": chexpert_5["macro_f1"],
        "chexpert_f1_micro": chexpert_14["micro_f1"],
        "chexpert_f1_macro": chexpert_14["macro_f1"],
        "chexpert_per_label_f1": chexpert_14["chexpert_per_label_f1"],
        "chexpert_label_names": chexpert_14["chexpert_label_names"],
        "chexpert_5_label_names": list(CHEXPERT_5_LABELS),
    }


def radgraph_f1(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, object]:
    if F1RadGraph is None:
        return {
            "radgraph_f1": None,
            "radgraph_f1_entity": None,
            "radgraph_f1_relation": None,
            "radgraph_available": False,
            "radgraph_error": repr(_RADGRAPH_IMPORT_ERROR),
        }

    real_makedirs = os.makedirs

    def safe_makedirs(path, mode=0o777, exist_ok=False):
        try:
            return real_makedirs(path, mode=mode, exist_ok=exist_ok)
        except FileExistsError:
            return None

    os.makedirs = safe_makedirs
    try:
        scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        mean_reward, *_ = scorer(hyps=list(predictions), refs=list(references))
    except Exception as exc:  # pragma: no cover - optional dependency/runtime path
        return {
            "radgraph_f1": None,
            "radgraph_f1_entity": None,
            "radgraph_f1_relation": None,
            "radgraph_available": False,
            "radgraph_error": repr(exc),
        }
    finally:
        os.makedirs = real_makedirs
    entity_f1, relation_f1, combined_f1 = mean_reward
    return {
        "radgraph_f1": float(combined_f1),
        "radgraph_f1_entity": float(entity_f1),
        "radgraph_f1_relation": float(relation_f1),
        "radgraph_available": True,
        "radgraph_error": None,
    }


def summarize_text_metrics(metric_values: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in metric_values.items()}


def default_metric_names() -> List[str]:
    return [
        "bleu_1",
        "bleu_4",
        "meteor",
        "rouge_l",
        "cider_d",
        "chexpert_f1_14_micro",
        "chexpert_f1_5_micro",
        "chexpert_f1_14_macro",
        "chexpert_f1_5_macro",
    ]


def evaluate_report_generation(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, object]:
    chexpert = chexpert_label_f1(predictions, references)
    metrics = {
        "bleu_1": corpus_bleu_1(predictions, references),
        "bleu_4": corpus_bleu_4(predictions, references),
        "meteor": meteor_score(predictions, references),
        "rouge_l": rouge_l(predictions, references),
        "cider_d": cider_d(predictions, references),
        "chexpert_f1_14_micro": chexpert["chexpert_f1_14_micro"],
        "chexpert_f1_5_micro": chexpert["chexpert_f1_5_micro"],
        "chexpert_f1_14_macro": chexpert["chexpert_f1_14_macro"],
        "chexpert_f1_5_macro": chexpert["chexpert_f1_5_macro"],
        "chexpert_f1_micro": chexpert["chexpert_f1_micro"],
        "chexpert_f1_macro": chexpert["chexpert_f1_macro"],
        "num_examples": len(predictions),
    }
    metrics.update(chexpert)
    return metrics
