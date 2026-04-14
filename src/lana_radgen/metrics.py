import os
import math
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import AutoConfig, AutoModel
from sklearn.metrics import classification_report

try:
    from huggingface_hub import hf_hub_download
except Exception as chexbert_hf_import_error:  # pragma: no cover - optional dependency path
    hf_hub_download = None
    _CHEXBERT_HF_IMPORT_ERROR = chexbert_hf_import_error
else:  # pragma: no cover - optional dependency path
    _CHEXBERT_HF_IMPORT_ERROR = None

try:
    from radgraph import F1RadGraph
except Exception as radgraph_import_error:  # pragma: no cover - optional dependency path
    F1RadGraph = None
    _RADGRAPH_IMPORT_ERROR = radgraph_import_error
else:  # pragma: no cover - optional dependency path
    _RADGRAPH_IMPORT_ERROR = None

try:
    from f1chexbert.f1chexbert import CACHE_DIR as _F1CHEXBERT_CACHE_DIR
except Exception as chexbert_import_error:  # pragma: no cover - optional dependency path
    _F1CHEXBERT_CACHE_DIR = None
    _CHEXBERT_IMPORT_ERROR = chexbert_import_error
else:  # pragma: no cover - optional dependency path
    _CHEXBERT_IMPORT_ERROR = None


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
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

RADGRAPH_MODEL_TYPE = "radgraph-xl"
RADGRAPH_TOKENIZER_MODEL_DIRNAME = "models--microsoft--BiomedVLP-CXR-BERT-general"
CHEXBERT_BERT_MODEL_DIRNAME = "models--bert-base-uncased"
CHEXBERT_REPO_ID = "StanfordAIMI/RRG_scorers"
CHEXBERT_WEIGHTS_FILENAME = "chexbert.pth"

_CHEXBERT_SCORER = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _path_has_files(path: Path | None) -> bool:
    if path is None or not path.exists() or not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except (OSError, StopIteration):
        return False
    return True


def _path_exists(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.exists()
    except OSError:
        return False


def _candidate_hf_cache_roots() -> List[Path]:
    candidates: List[Path] = []
    env_hf_home = os.getenv("HF_HOME")
    env_hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    env_local_app_data = os.getenv("LOCALAPPDATA")

    if env_hub_cache:
        candidates.append(Path(env_hub_cache))
    if env_hf_home:
        candidates.append(Path(env_hf_home) / "hub")
    if env_local_app_data:
        candidates.append(Path(env_local_app_data) / "huggingface" / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized not in seen:
            unique_candidates.append(candidate)
            seen.add(normalized)
    return unique_candidates


def _resolve_hf_snapshot_dir(model_dirname: str) -> Path | None:
    for cache_root in [Path(__file__).resolve().parents[2] / ".cache" / "huggingface" / "hub", *_candidate_hf_cache_roots()]:
        model_root = cache_root / model_dirname
        if not _path_has_files(model_root):
            continue
        snapshots_dir = model_root / "snapshots"
        if _path_has_files(snapshots_dir):
            for snapshot_dir in sorted(snapshots_dir.iterdir(), reverse=True):
                if _path_has_files(snapshot_dir):
                    return snapshot_dir
        return model_root
    return None


def _resolve_radgraph_runtime_paths(model_type: str = RADGRAPH_MODEL_TYPE) -> tuple[Path, Path | None, bool]:
    env_model_cache_dir = os.getenv("LANA_RADGRAPH_MODEL_CACHE_DIR")
    env_tokenizer_cache_dir = os.getenv("LANA_RADGRAPH_TOKENIZER_CACHE_DIR")
    env_hf_cache_dir = os.getenv("LANA_HF_CACHE_DIR")
    repo_root = Path(__file__).resolve().parents[2]
    workspace_model_cache_dir = repo_root / ".cache" / "radgraph"
    workspace_tokenizer_cache_dir = repo_root / ".cache" / "huggingface" / "hub"

    if env_model_cache_dir:
        model_cache_dir = Path(env_model_cache_dir)
    else:
        model_cache_dir = workspace_model_cache_dir

    tokenizer_cache_candidates: List[Path] = []
    if env_tokenizer_cache_dir:
        tokenizer_cache_candidates.append(Path(env_tokenizer_cache_dir))
    if env_hf_cache_dir:
        tokenizer_cache_candidates.append(Path(env_hf_cache_dir))
    tokenizer_cache_candidates.append(workspace_tokenizer_cache_dir)
    tokenizer_cache_candidates.extend(_candidate_hf_cache_roots())

    tokenizer_cache_dir = None
    tokenizer_ready = False
    for candidate in tokenizer_cache_candidates:
        if _path_has_files(candidate / RADGRAPH_TOKENIZER_MODEL_DIRNAME):
            tokenizer_cache_dir = candidate
            tokenizer_ready = True
            break

    model_ready = _path_has_files(model_cache_dir / model_type)
    offline_ready = model_ready and tokenizer_ready
    return model_cache_dir, tokenizer_cache_dir, offline_ready


def _resolve_chexbert_runtime_paths() -> tuple[str | Path, str | Path, Path, bool]:
    env_tokenizer_dir = os.getenv("LANA_CHEXPERT_TOKENIZER_DIR")
    env_config_dir = os.getenv("LANA_CHEXPERT_CONFIG_DIR")
    env_weights_path = os.getenv("LANA_CHEXPERT_WEIGHTS_PATH")
    repo_root = Path(__file__).resolve().parents[2]

    tokenizer_dir = Path(env_tokenizer_dir) if env_tokenizer_dir else _resolve_hf_snapshot_dir(CHEXBERT_BERT_MODEL_DIRNAME)
    config_dir = Path(env_config_dir) if env_config_dir else tokenizer_dir

    candidate_weight_paths: List[Path] = []
    if env_weights_path:
        candidate_weight_paths.append(Path(env_weights_path))
    candidate_weight_paths.append(repo_root / ".cache" / "chexbert" / CHEXBERT_WEIGHTS_FILENAME)
    if _F1CHEXBERT_CACHE_DIR:
        candidate_weight_paths.append(Path(_F1CHEXBERT_CACHE_DIR) / CHEXBERT_WEIGHTS_FILENAME)

    weights_path = candidate_weight_paths[0]
    for candidate in candidate_weight_paths:
        if _path_exists(candidate):
            weights_path = candidate
            break

    tokenizer_source: str | Path = tokenizer_dir if tokenizer_dir is not None else "bert-base-uncased"
    config_source: str | Path = config_dir if config_dir is not None else tokenizer_source
    offline_ready = tokenizer_dir is not None and _path_exists(weights_path)
    return tokenizer_source, config_source, weights_path, offline_ready


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    original_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text))


def _generate_attention_masks(batch: torch.Tensor, source_lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)


def _tokenize_reports_for_chexbert(reports: Sequence[str], tokenizer: BertTokenizer) -> List[List[int]]:
    tokenized_reports: List[List[int]] = []
    for report in reports:
        normalized = re.sub(r"\s+", " ", (report or "").strip().replace("\n", " "))
        tokenized = tokenizer.tokenize(normalized)
        if tokenized:
            token_ids = tokenizer.convert_tokens_to_ids(tokenized)
            encoded = tokenizer.build_inputs_with_special_tokens(token_ids)
            if len(encoded) > 512:
                encoded = encoded[:511] + [tokenizer.sep_token_id]
            tokenized_reports.append(encoded)
        else:
            tokenized_reports.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return tokenized_reports


class _CheXbertLabeler(nn.Module):
    def __init__(self, config_source: str | Path, local_files_only: bool):
        super().__init__()
        config = AutoConfig.from_pretrained(str(config_source), local_files_only=local_files_only)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        return [head(cls_hidden) for head in self.linear_heads]


class _CheXbertScorer:
    def __init__(self, device=None):
        if _CHEXBERT_IMPORT_ERROR is not None:
            raise RuntimeError(f"f1chexbert import unavailable: {_CHEXBERT_IMPORT_ERROR!r}")

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)

        tokenizer_source, config_source, weights_path, offline_ready = _resolve_chexbert_runtime_paths()
        local_files_only = offline_ready or isinstance(tokenizer_source, Path)
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_source), local_files_only=local_files_only)
        self.model = _CheXbertLabeler(config_source=config_source, local_files_only=local_files_only)

        checkpoint_path = weights_path
        if not _path_exists(checkpoint_path):
            if hf_hub_download is None:
                raise RuntimeError(f"CheXbert weights unavailable and huggingface_hub import failed: {_CHEXBERT_HF_IMPORT_ERROR!r}")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=CHEXBERT_REPO_ID,
                    filename=CHEXBERT_WEIGHTS_FILENAME,
                    local_dir=str(checkpoint_path.parent),
                    local_dir_use_symlinks=False,
                )
            )

        state_dict = torch.load(checkpoint_path, map_location=self.device)["model_state_dict"]
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_state_dict[key.replace("module.", "")] = value

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model = self.model.to(self.device).eval()
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def label_reports(self, reports: Sequence[str]) -> List[List[int]]:
        encoded_reports = _tokenize_reports_for_chexbert(reports, self.tokenizer)
        labels: List[List[int]] = []
        for encoded in encoded_reports:
            batch = torch.LongTensor([encoded])
            source_lengths = [len(encoded)]
            attention_mask = _generate_attention_masks(batch, source_lengths, self.device)
            outputs = self.model(batch.to(self.device), attention_mask)
            classes = [head.argmax(dim=1).item() for head in outputs]
            label_vector = []
            for predicted_class in classes:
                if predicted_class == 0:
                    label_vector.append(0)
                elif predicted_class == 2:
                    label_vector.append(0)
                elif predicted_class == 1:
                    label_vector.append(1)
                elif predicted_class == 3:
                    label_vector.append(1)
                else:
                    label_vector.append(0)
            labels.append(label_vector)
        return labels


def _get_chexbert_scorer() -> _CheXbertScorer:
    global _CHEXBERT_SCORER
    if _CHEXBERT_SCORER is None:
        _CHEXBERT_SCORER = _CheXbertScorer()
    return _CHEXBERT_SCORER


def _compute_chexpert_metrics_from_label_vectors(
    predicted_label_vectors: Sequence[Sequence[int]],
    reference_label_vectors: Sequence[Sequence[int]],
) -> Dict[str, object]:
    report_14 = classification_report(
        reference_label_vectors,
        predicted_label_vectors,
        target_names=CHEXPERT_14_LABELS,
        output_dict=True,
        zero_division=0,
    )
    chexpert_5_indices = [CHEXPERT_14_LABELS.index(label) for label in CHEXPERT_5_LABELS]
    reference_label_vectors_5 = [[row[idx] for idx in chexpert_5_indices] for row in reference_label_vectors]
    predicted_label_vectors_5 = [[row[idx] for idx in chexpert_5_indices] for row in predicted_label_vectors]
    report_5 = classification_report(
        reference_label_vectors_5,
        predicted_label_vectors_5,
        target_names=CHEXPERT_5_LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_label_f1 = {label: float(report_14[label]["f1-score"]) for label in CHEXPERT_14_LABELS}
    return {
        "chexpert_f1_14_micro": float(report_14["micro avg"]["f1-score"]),
        "chexpert_f1_14_macro": float(report_14["macro avg"]["f1-score"]),
        "chexpert_f1_5_micro": float(report_5["micro avg"]["f1-score"]),
        "chexpert_f1_5_macro": float(report_5["macro avg"]["f1-score"]),
        "chexpert_f1_micro": float(report_14["micro avg"]["f1-score"]),
        "chexpert_f1_macro": float(report_14["macro avg"]["f1-score"]),
        "chexpert_per_label_f1": per_label_f1,
        "chexpert_label_names": list(CHEXPERT_14_LABELS),
        "chexpert_5_label_names": list(CHEXPERT_5_LABELS),
        "chexpert_available": True,
        "chexpert_error": None,
    }


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
    try:
        scorer = _get_chexbert_scorer()
        label_vectors = scorer.label_reports(references)
    except Exception:
        return []
    return [
        {label: int(label_vector[idx]) for idx, label in enumerate(CHEXPERT_14_LABELS)}
        for label_vector in label_vectors
    ]


def chexpert_label_f1(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, object]:
    try:
        scorer = _get_chexbert_scorer()
        predicted_label_vectors = scorer.label_reports(predictions)
        reference_label_vectors = scorer.label_reports(references)
        return _compute_chexpert_metrics_from_label_vectors(predicted_label_vectors, reference_label_vectors)
    except Exception as exc:
        return {
            "chexpert_f1_14_micro": None,
            "chexpert_f1_14_macro": None,
            "chexpert_f1_5_micro": None,
            "chexpert_f1_5_macro": None,
            "chexpert_f1_micro": None,
            "chexpert_f1_macro": None,
            "chexpert_per_label_f1": {label: None for label in CHEXPERT_14_LABELS},
            "chexpert_label_names": list(CHEXPERT_14_LABELS),
            "chexpert_5_label_names": list(CHEXPERT_5_LABELS),
            "chexpert_available": False,
            "chexpert_error": repr(exc),
        }


def chexpert_label_f1_from_reference_labels(predictions: Sequence[str], reference_labels: Sequence[Dict[str, int]]) -> Dict[str, object]:
    try:
        scorer = _get_chexbert_scorer()
        predicted_label_vectors = scorer.label_reports(predictions)
        reference_label_vectors = [
            [int(reference_label_map.get(label, 0)) for label in CHEXPERT_14_LABELS]
            for reference_label_map in reference_labels
        ]
        return _compute_chexpert_metrics_from_label_vectors(predicted_label_vectors, reference_label_vectors)
    except Exception as exc:
        return {
            "chexpert_f1_14_micro": None,
            "chexpert_f1_14_macro": None,
            "chexpert_f1_5_micro": None,
            "chexpert_f1_5_macro": None,
            "chexpert_f1_micro": None,
            "chexpert_f1_macro": None,
            "chexpert_per_label_f1": {label: None for label in CHEXPERT_14_LABELS},
            "chexpert_label_names": list(CHEXPERT_14_LABELS),
            "chexpert_5_label_names": list(CHEXPERT_5_LABELS),
            "chexpert_available": False,
            "chexpert_error": repr(exc),
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

    model_cache_dir, tokenizer_cache_dir, offline_ready = _resolve_radgraph_runtime_paths(model_type=RADGRAPH_MODEL_TYPE)
    env_overrides = {}
    if offline_ready:
        env_overrides["HF_HUB_OFFLINE"] = "1"
        env_overrides["TRANSFORMERS_OFFLINE"] = "1"

    os.makedirs = safe_makedirs
    try:
        with _temporary_env(env_overrides):
            scorer = F1RadGraph(
                reward_level="all",
                model_type=RADGRAPH_MODEL_TYPE,
                model_cache_dir=str(model_cache_dir),
                tokenizer_cache_dir=str(tokenizer_cache_dir) if tokenizer_cache_dir is not None else None,
            )
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
