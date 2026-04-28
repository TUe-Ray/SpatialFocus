import ast
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

try:
    from loguru import logger as eval_logger
except ModuleNotFoundError:
    class _FallbackLogger:
        def info(self, message):
            print(message)

        def warning(self, message):
            print(message)

    eval_logger = _FallbackLogger()

LETTERS = ["A", "B", "C", "D"]
DEFAULT_NUM_SAMPLES = 100
DEFAULT_SAMPLE_SEED = 42
DEFAULT_OPTION_SHUFFLE_SEED = 0

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

PROMPT_TEMPLATES = {
    "option_shuffle_v1": "\n\n".join(
        [
            "These are frames of a video.",
            "Question:\n{question}",
            "Options:\n{options}",
            "Answer with the option's letter from the given choices directly.",
        ]
    ),
    "evidence_json_mca_v1": "\n\n".join(
        [
            "These are frames of a video.",
            "Question:\n{question}",
            "Options:\n{options}",
            'Answer in this JSON format:\n{{\n  "answer": "A/B/C/D",\n  "evidence_objects": ["object1", "object2"],\n  "spatial_evidence": "one short sentence describing the visible spatial relation",\n  "uncertainty": "low/medium/high"\n}}',
            "Rules:\n- The answer must be one of A, B, C, or D.\n- evidence_objects should only include objects that are visible or relevant to the question.\n- spatial_evidence should be a short observable statement, not a step-by-step reasoning chain.\n- Do not include any other text outside the JSON.",
        ]
    ),
    "evidence_json_numeric_v1": "\n\n".join(
        [
            "These are frames of a video.",
            "Question:\n{question}",
            'Answer in this JSON format:\n{{\n  "answer": "<number or short phrase>",\n  "visible_reference": "what object, surface, or boundary you used as reference",\n  "estimate_reason": "one short observable cue",\n  "uncertainty": "low/medium/high"\n}}',
            "Rules:\n- The answer should be concise.\n- visible_reference should mention what visual reference was used.\n- estimate_reason should be a short observable cue, not a step-by-step reasoning chain.\n- Do not include any other text outside the JSON.",
        ]
    ),
}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        eval_logger.warning(f"Ignoring invalid integer value for {name}: {value!r}")
        return default


def _prompt_variant():
    return os.getenv("PROMPT_VARIANT", "option_shuffle").strip() or "option_shuffle"


def _option_shuffle_seeds():
    seeds = os.getenv("OPTION_SHUFFLE_SEEDS")
    if not seeds:
        seeds = os.getenv("OPTION_SHUFFLE_SEED")
    if not seeds:
        return [DEFAULT_OPTION_SHUFFLE_SEED]
    parsed = []
    for item in re.split(r"[, ]+", seeds.strip()):
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            eval_logger.warning(f"Ignoring invalid option shuffle seed: {item!r}")
    return parsed or [DEFAULT_OPTION_SHUFFLE_SEED]


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_template_sha256(template_name):
    return _sha256_text(PROMPT_TEMPLATES[template_name])


def _stable_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _plain_doc(doc):
    return {key: doc[key] for key in doc.keys()}


def _clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_options(options):
    if options is None:
        return {}
    parsed = {}
    for idx, raw in enumerate(options):
        raw_text = _clean_text(raw)
        match = re.match(r"^\s*([A-Z])\s*[\.\):\-]\s*(.*)\s*$", raw_text)
        if match:
            letter, text = match.group(1).upper(), match.group(2).strip()
        else:
            if idx >= len(LETTERS):
                continue
            letter, text = LETTERS[idx], raw_text
        if letter in LETTERS:
            parsed[letter] = text
    return parsed


def format_options(options_dict):
    return [f"{letter}. {options_dict[letter]}" for letter in LETTERS if letter in options_dict]


def stable_sample_id(doc, dataset_index):
    existing_id = doc.get("id")
    if existing_id not in (None, ""):
        dataset_name = _clean_text(doc.get("dataset")) or "vsibench"
        return f"{dataset_name}_{existing_id}"

    payload = {
        "scene_name": doc.get("scene_name"),
        "question_type": doc.get("question_type"),
        "question": doc.get("question"),
        "options": doc.get("options"),
        "ground_truth": doc.get("ground_truth"),
        "dataset_index": dataset_index,
    }
    short_hash = _sha256_text(_stable_json(payload))[:16]
    scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", _clean_text(doc.get("scene_name")) or "scene")
    question_type = re.sub(r"[^A-Za-z0-9_.-]+", "_", _clean_text(doc.get("question_type")) or "question")
    return f"{scene}_{question_type}_{dataset_index}_{short_hash}"


def _selection_key(sample_seed, row):
    return (_sha256_text(f"{sample_seed}:{row['sample_id']}"), row["dataset_index"])


def _is_mca(row):
    return row.get("question_type") in MCA_QUESTION_TYPES and bool(parse_options(row.get("options")))


def select_probe_rows(dataset, prompt_variant, num_samples, sample_seed):
    rows = []
    for dataset_index, doc in enumerate(dataset):
        row = _plain_doc(doc)
        row["dataset_index"] = dataset_index
        row["sample_id"] = stable_sample_id(row, dataset_index)
        rows.append(row)

    sample_id_counts = Counter(row["sample_id"] for row in rows)
    for row in rows:
        if sample_id_counts[row["sample_id"]] > 1:
            row["sample_id"] = f"{row['sample_id']}_{row['dataset_index']}"

    if prompt_variant == "option_shuffle":
        rows = [row for row in rows if _is_mca(row)]

    rows.sort(key=lambda row: _selection_key(sample_seed, row))
    if num_samples is not None and num_samples > 0:
        rows = rows[:num_samples]

    selected = []
    for probe_index, row in enumerate(rows):
        row = dict(row)
        row["probe_index"] = probe_index
        selected.append(row)
    return selected


def shuffle_options_for_sample(options_dict, sample_id, option_shuffle_seed):
    original_letters = [letter for letter in LETTERS if letter in options_dict]
    ranked = sorted(original_letters, key=lambda letter: _sha256_text(f"{option_shuffle_seed}:{sample_id}:{letter}:{options_dict[letter]}"))
    if ranked == original_letters and len(ranked) > 1:
        ranked = ranked[1:] + ranked[:1]
    presented_letters = original_letters
    presented_options = {presented: options_dict[original] for presented, original in zip(presented_letters, ranked)}
    presented_to_original = {presented: original for presented, original in zip(presented_letters, ranked)}
    original_to_presented = {original: presented for presented, original in presented_to_original.items()}
    return presented_options, presented_to_original, original_to_presented


def _resolve_gt_letter(options_dict, ground_truth):
    gt = _clean_text(ground_truth).upper()
    if gt in options_dict:
        return gt
    normalized_gt = _clean_text(ground_truth).lower()
    for letter, text in options_dict.items():
        if _clean_text(text).lower() == normalized_gt:
            return letter
    return None


def _base_probe_fields(row, prompt_variant):
    options_dict = parse_options(row.get("options"))
    gt_letter = _resolve_gt_letter(options_dict, row.get("ground_truth"))
    gt_answer_text = options_dict.get(gt_letter) if gt_letter else _clean_text(row.get("ground_truth"))
    base = dict(row)
    base.update(
        {
            "prompt_variant": prompt_variant,
            "original_options": options_dict,
            "gt_original_letter": gt_letter,
            "gt_answer_text": gt_answer_text,
            "options": format_options(options_dict) if options_dict else row.get("options", []),
        }
    )
    return base


def _expand_option_shuffle_rows(selected, option_shuffle_seeds):
    expanded = []
    for row in selected:
        base = _base_probe_fields(row, "option_shuffle")
        options_dict = base["original_options"]
        gt_original_letter = base["gt_original_letter"]
        for option_shuffle_seed in option_shuffle_seeds:
            presented_options, presented_to_original, original_to_presented = shuffle_options_for_sample(options_dict, base["sample_id"], option_shuffle_seed)
            gt_presented_letter = original_to_presented.get(gt_original_letter)
            expanded_row = dict(base)
            expanded_row.update(
                {
                    "option_shuffle_seed": option_shuffle_seed,
                    "presented_options": presented_options,
                    "presented_to_original": presented_to_original,
                    "original_to_presented": original_to_presented,
                    "gt_presented_letter": gt_presented_letter,
                    "options": format_options(presented_options),
                    "ground_truth": gt_presented_letter or gt_original_letter or row.get("ground_truth"),
                }
            )
            expanded.append(expanded_row)
    return expanded


def _expand_evidence_rows(selected):
    expanded = []
    for row in selected:
        base = _base_probe_fields(row, "evidence_json")
        base.update(
            {
                "option_shuffle_seed": None,
                "presented_options": {},
                "presented_to_original": {},
                "original_to_presented": {},
                "gt_presented_letter": None,
            }
        )
        expanded.append(base)
    return expanded


def _is_primary_process():
    for key in ("RANK", "LOCAL_RANK", "PROCESS_INDEX"):
        value = os.getenv(key)
        if value not in (None, "", "0"):
            return False
    return True


def _write_selected_samples(selected, prompt_variant, num_samples, sample_seed, option_shuffle_seeds):
    output_dir = os.getenv("VSIBENCH_PROBE_OUTPUT_DIR")
    if not output_dir or not _is_primary_process():
        return
    payload = {
        "dataset": "vsibench",
        "prompt_variant": prompt_variant,
        "num_samples": num_samples,
        "actual_num_samples": len(selected),
        "sample_seed": sample_seed,
        "option_shuffle_seeds": option_shuffle_seeds if prompt_variant == "option_shuffle" else None,
        "sample_order": [
            {
                "probe_index": row["probe_index"],
                "sample_id": row["sample_id"],
                "dataset_index": row["dataset_index"],
                "scene_name": row.get("scene_name"),
                "question_type": row.get("question_type"),
            }
            for row in selected
        ],
    }
    path = Path(output_dir) / "selected_samples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_docs(dataset):
    from datasets import Dataset

    prompt_variant = _prompt_variant()
    if prompt_variant not in {"option_shuffle", "evidence_json"}:
        raise ValueError(f"Unsupported VSiBench probe prompt variant: {prompt_variant}")

    num_samples = _env_int("NUM_SAMPLES", DEFAULT_NUM_SAMPLES)
    sample_seed = _env_int("SAMPLE_SEED", DEFAULT_SAMPLE_SEED)
    option_shuffle_seeds = _option_shuffle_seeds()
    if num_samples <= 0:
        raise ValueError(f"NUM_SAMPLES must be a positive integer for VSiBench probe runs, got {num_samples}")

    selected = select_probe_rows(dataset, prompt_variant, num_samples, sample_seed)
    _write_selected_samples(selected, prompt_variant, num_samples, sample_seed, option_shuffle_seeds)

    if prompt_variant == "option_shuffle":
        expanded = _expand_option_shuffle_rows(selected, option_shuffle_seeds)
    else:
        expanded = _expand_evidence_rows(selected)

    eval_logger.info(
        f"Prepared VSiBench probe dataset: variant={prompt_variant}, base_samples={len(selected)}, rows={len(expanded)}, sample_seed={sample_seed}, option_shuffle_seeds={option_shuffle_seeds}"
    )
    return Dataset.from_list(expanded)


def _format_options_block(options):
    return "\n".join(options)


def _prompt_template_name_for_doc(doc):
    prompt_variant = doc.get("prompt_variant") or _prompt_variant()
    if prompt_variant == "option_shuffle":
        return "option_shuffle_v1"
    if doc.get("question_type") in MCA_QUESTION_TYPES and (doc.get("options") or []):
        return "evidence_json_mca_v1"
    return "evidence_json_numeric_v1"


def prompt_template_metadata_for_doc(doc):
    template_name = _prompt_template_name_for_doc(doc)
    return template_name, prompt_template_sha256(template_name)


def render_prompt(doc):
    template_name = _prompt_template_name_for_doc(doc)
    return PROMPT_TEMPLATES[template_name].format(
        question=_clean_text(doc.get("question")),
        options=_format_options_block(doc.get("options") or []),
    )


def vsibench_probe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return render_prompt(doc)


def vsibench_probe_doc_to_target(doc):
    if doc.get("prompt_variant") == "option_shuffle":
        return doc.get("gt_presented_letter") or doc.get("gt_original_letter") or doc.get("ground_truth")
    return doc.get("ground_truth")


def _contains_placeholder_option_list(text, valid_letters):
    normalized = _clean_text(text).upper()
    compact = re.sub(r"\s+", "", normalized)
    expected = "/".join(valid_letters)
    if compact == expected:
        return True

    letter_list = re.findall(rf"\b([{''.join(re.escape(letter) for letter in valid_letters)}])\b", normalized)
    distinct = list(dict.fromkeys(letter_list))
    if len(set(distinct)) > 1:
        return True
    return False


def parse_option_letter(text, valid_letters=None):
    valid_letters = valid_letters or LETTERS
    valid_letters = [letter.upper() for letter in valid_letters]
    letter_class = "".join(re.escape(letter) for letter in valid_letters)
    raw = _clean_text(text)
    stripped = raw.strip()
    compact = stripped.strip("()[]{} \t\r\n.:,;!")

    if len(compact) == 1 and compact.upper() in valid_letters:
        return compact.upper(), None

    json_answer_pattern = re.compile(rf"""["']answer["']\s*:\s*["']\s*([{letter_class}])\s*["']""", re.IGNORECASE)
    match = json_answer_pattern.search(stripped[:300])
    if match:
        return match.group(1).upper(), None

    if _contains_placeholder_option_list(stripped, valid_letters):
        return None, "Ambiguous or placeholder option-letter answer"

    answer_pattern = re.compile(rf"\b(?:final\s+answer|answer)\s*(?:is|:|-)?\s*\(?([{letter_class}])\)?\b", re.IGNORECASE)
    match = answer_pattern.search(stripped[:200])
    if match:
        return match.group(1).upper(), None

    head = stripped[:80]
    match = re.search(rf"(?<![A-Za-z0-9])\(?([{letter_class}])\)?(?![A-Za-z0-9])", head, re.IGNORECASE)
    if match:
        return match.group(1).upper(), None

    return None, "Could not extract valid option letter"


def _extract_json_candidate(text):
    raw = _clean_text(text)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1)

    start = raw.find("{")
    if start < 0:
        return raw

    in_string = False
    escape = False
    depth = 0
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return raw[start:]


def parse_json_response(text):
    candidate = _extract_json_candidate(text)
    attempts = [candidate]
    attempts.append(re.sub(r",\s*([}\]])", r"\1", candidate))

    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass

    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed, None
    except Exception as exc:
        return None, f"Invalid JSON: {exc}"

    return None, "Invalid JSON: parsed value is not an object"


def _normalize_uncertainty(value):
    uncertainty = _clean_text(value).lower()
    if uncertainty in {"low", "medium", "high"}:
        return uncertainty
    return uncertainty or None


def _as_string_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [_clean_text(value)] if _clean_text(value) else []


def _first_float(value):
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", _clean_text(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if pred is None or target in (None, 0):
        return None
    num_pts = int((end - start) / interval + 2)
    thresholds = [start + interval * idx for idx in range(num_pts - 1)]
    thresholds.append(end)
    rel_error = abs(pred - target) / abs(target)
    return sum(rel_error <= 1 - threshold for threshold in thresholds) / len(thresholds)


def _numeric_metrics(pred, target):
    metrics = {
        "numeric_pred": pred,
        "numeric_target": target,
        "numeric_abs_error": None,
        "numeric_rel_error": None,
        "numeric_mra": None,
        "numeric_within_10pct": None,
        "numeric_within_25pct": None,
        "numeric_exact_match": None,
    }
    if pred is None or target is None:
        return metrics

    abs_error = abs(pred - target)
    rel_error = None if target == 0 else abs_error / abs(target)
    metrics.update(
        {
            "numeric_abs_error": abs_error,
            "numeric_rel_error": rel_error,
            "numeric_mra": _mean_relative_accuracy(pred, target),
            "numeric_within_10pct": None if rel_error is None else rel_error <= 0.10,
            "numeric_within_25pct": None if rel_error is None else rel_error <= 0.25,
            "numeric_exact_match": pred == target,
        }
    )
    return metrics


def _fallback_short_answer(text):
    raw = _clean_text(text)
    if not raw:
        return None
    for pattern in [
        r"\b(?:final\s+answer|answer)\s*(?:is|:|-)\s*(.+)",
        r"""["']answer["']\s*:\s*["']?([^"',}\n]+)""",
    ]:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().strip("`\"' .,;")
            return answer[:80] if answer else None
    return raw.splitlines()[0].strip()[:80]


def _normalize_answer_text(value):
    return " ".join(_clean_text(value).lower().split())


def _build_common_record(doc, raw_prediction):
    prompt_template_name, prompt_template_hash = prompt_template_metadata_for_doc(doc)
    probe_uid = f"{doc.get('sample_id')}::evidence_json"
    if doc.get("prompt_variant") == "option_shuffle":
        probe_uid = f"{doc.get('sample_id')}::shuffle_seed={doc.get('option_shuffle_seed')}"

    record = {
        "probe_uid": probe_uid,
        "probe_index": doc.get("probe_index"),
        "sample_id": doc.get("sample_id"),
        "dataset_index": doc.get("dataset_index"),
        "scene_name": doc.get("scene_name"),
        "question_type": doc.get("question_type"),
        "question": doc.get("question"),
        "prompt_variant": doc.get("prompt_variant"),
        "prompt_template_name": prompt_template_name,
        "prompt_template_sha256": prompt_template_hash,
        "model_raw_prediction": raw_prediction,
    }
    if os.getenv("SAVE_RENDERED_PROMPTS", "").strip().lower() in {"1", "true", "yes", "y"}:
        rendered_prompt = render_prompt(doc)
        record["rendered_prompt"] = rendered_prompt
        record["rendered_prompt_sha256"] = _sha256_text(rendered_prompt)
    return record


def _process_option_shuffle(doc, raw_prediction):
    presented_options = dict(doc.get("presented_options") or {})
    original_options = dict(doc.get("original_options") or {})
    presented_to_original = dict(doc.get("presented_to_original") or {})
    original_to_presented = dict(doc.get("original_to_presented") or {})
    valid_letters = [letter for letter in LETTERS if letter in presented_options]
    model_presented_letter, parse_error = parse_option_letter(raw_prediction, valid_letters)
    model_original_letter = presented_to_original.get(model_presented_letter) if model_presented_letter else None
    model_answer_text = original_options.get(model_original_letter) if model_original_letter else None
    gt_original_letter = doc.get("gt_original_letter")
    gt_presented_letter = doc.get("gt_presented_letter")

    record = _build_common_record(doc, raw_prediction)
    record.update(
        {
            "option_shuffle_seed": doc.get("option_shuffle_seed"),
            "original_options": original_options,
            "presented_options": presented_options,
            "presented_to_original": presented_to_original,
            "original_to_presented": original_to_presented,
            "gt_original_letter": gt_original_letter,
            "gt_presented_letter": gt_presented_letter,
            "gt_answer_text": doc.get("gt_answer_text"),
            "model_presented_letter": model_presented_letter,
            "model_original_letter": model_original_letter,
            "model_answer_text": model_answer_text,
            "correct_original_space": bool(model_original_letter == gt_original_letter) if model_original_letter and gt_original_letter else False,
            "correct_presented_space": bool(model_presented_letter == gt_presented_letter) if model_presented_letter and gt_presented_letter else False,
            "parse_ok": parse_error is None,
            "parse_error": parse_error,
        }
    )
    return record


def _process_evidence_json(doc, raw_prediction):
    options = dict(doc.get("original_options") or {})
    valid_letters = [letter for letter in LETTERS if letter in options]
    gt_letter = doc.get("gt_original_letter")
    gt_answer_text = doc.get("gt_answer_text")
    parsed, parse_error = parse_json_response(raw_prediction)

    model_answer = None
    model_answer_text = None
    evidence_objects = []
    spatial_evidence = None
    visible_reference = None
    estimate_reason = None
    uncertainty = None
    answer_parse_ok = False
    answer_parse_error = None
    correct = None
    open_ended_normalized_match = None
    numeric_target = _first_float(doc.get("ground_truth"))
    is_mca = doc.get("question_type") in MCA_QUESTION_TYPES and bool(options)
    is_numeric = not is_mca and numeric_target is not None
    numeric_fields = _numeric_metrics(None, numeric_target)

    if parsed is not None:
        answer_source = parsed.get("answer")
        uncertainty = _normalize_uncertainty(parsed.get("uncertainty"))
        evidence_objects = _as_string_list(parsed.get("evidence_objects"))
        spatial_evidence = _clean_text(parsed.get("spatial_evidence")) or None
        visible_reference = _clean_text(parsed.get("visible_reference")) or None
        estimate_reason = _clean_text(parsed.get("estimate_reason")) or None
    else:
        answer_source = raw_prediction

    if is_mca:
        model_letter, answer_parse_error = parse_option_letter(answer_source, valid_letters)
        answer_parse_ok = answer_parse_error is None
        if answer_parse_ok:
            model_answer = model_letter or model_answer
            model_answer_text = options.get(model_letter) if model_letter else None
            correct = bool(model_letter == gt_letter) if model_letter and gt_letter else False
        else:
            model_answer = _clean_text(answer_source) or None
            model_answer_text = None
            correct = False
    else:
        answer_text = _clean_text(answer_source)
        if not answer_text and parsed is None:
            answer_text = _fallback_short_answer(raw_prediction) or ""
        if not answer_text:
            answer_parse_error = "Missing answer"
            answer_parse_ok = False
        else:
            if parsed is None:
                fallback = _fallback_short_answer(raw_prediction)
                answer_text = fallback or answer_text
            model_answer = answer_text
            model_answer_text = answer_text
            pred_float = _first_float(answer_text)
            if is_numeric:
                answer_parse_ok = pred_float is not None
                answer_parse_error = None if answer_parse_ok else "Could not extract numeric answer"
                numeric_fields = _numeric_metrics(pred_float, numeric_target)
                correct = None
            else:
                answer_parse_ok = True
                answer_parse_error = None
                open_ended_normalized_match = _normalize_answer_text(answer_text) == _normalize_answer_text(doc.get("ground_truth"))
                correct = open_ended_normalized_match

    record = _build_common_record(doc, raw_prediction)
    record.update(
        {
            "options": options,
            "gt_letter": gt_letter,
            "gt_answer_text": gt_answer_text,
            "is_mca": is_mca,
            "is_numeric": is_numeric,
            "model_answer": model_answer,
            "model_answer_text": model_answer_text,
            "evidence_objects": evidence_objects,
            "spatial_evidence": spatial_evidence,
            "visible_reference": visible_reference,
            "estimate_reason": estimate_reason,
            "uncertainty": uncertainty,
            "json_parse_ok": parsed is not None,
            "json_parse_error": parse_error,
            "answer_parse_ok": answer_parse_ok,
            "answer_parse_error": answer_parse_error,
            "correct": correct,
            "open_ended_normalized_match": open_ended_normalized_match,
        }
    )
    record.update(numeric_fields)
    return record


def vsibench_probe_process_results(doc, results):
    raw_prediction = results[0] if results else ""
    if doc.get("prompt_variant") == "option_shuffle":
        record = _process_option_shuffle(doc, raw_prediction)
    elif doc.get("prompt_variant") == "evidence_json":
        record = _process_evidence_json(doc, raw_prediction)
    else:
        record = _build_common_record(doc, raw_prediction)
        record.update({"parse_ok": False, "parse_error": f"Unknown prompt variant: {doc.get('prompt_variant')}"})
    return {"vsibench_probe_score": record}


def vsibench_probe_aggregate_results(results):
    if not results:
        return 0.0
    variant = results[0].get("prompt_variant")
    if variant == "option_shuffle":
        return 100.0 * sum(row.get("correct_original_space") is True for row in results) / len(results)
    if variant == "evidence_json":
        mca_rows = [row for row in results if row.get("is_mca")]
        if mca_rows:
            return 100.0 * sum(row.get("correct") is True for row in mca_rows) / len(mca_rows)
        numeric_mras = [row.get("numeric_mra") for row in results if isinstance(row.get("numeric_mra"), (int, float))]
        if numeric_mras:
            return 100.0 * sum(numeric_mras) / len(numeric_mras)
        return 100.0 * sum(row.get("correct") is True for row in results) / len(results)
    return 0.0


def evidence_object_counts(records, wrong_only=False):
    counter = Counter()
    for record in records:
        if wrong_only and record.get("correct") is not False:
            continue
        counter.update(record.get("evidence_objects") or [])
    return counter
