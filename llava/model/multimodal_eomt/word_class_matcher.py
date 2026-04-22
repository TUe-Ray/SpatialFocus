"""Utilities for matching question-selected words to COCO panoptic classes."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Things: indices 0-79  (80 COCO instance categories)
# Stuff:  indices 80-132 (53 COCO stuff categories)
COCO_PANOPTIC_CLASS_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
    "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door",
    "floor_wood", "flower", "fruit", "gravel", "house", "light", "mirror",
    "net", "pillow", "platform", "playingfield", "railroad", "river", "road",
    "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "tree",
    "wall_brick", "wall_concrete", "wall_panel", "wall_stone", "wall_tile",
    "wall_wood", "water", "window_blind", "window", "pavement", "mountain",
    "grass", "dirt", "paper", "food", "building", "rock", "wall", "ceiling",
    "textile", "fence", "sky", "ground",
]

COCO_WORD_CLASS_ALIASES: Dict[str, str] = {
    "bag": "backpack",
    "blinds": "window_blind",
    "ceiling lamp": "light",
    "computer mouse": "mouse",
    "lamp": "light",
    "monitor": "tv",
    "office chair": "chair",
    "plant": "potted_plant",
    "plant pot": "potted_plant",
    "sofa": "couch",
    "stove": "oven",
    "table": "dining_table",
    "table lamp": "light",
    "tap": "sink",
    "telephone": "cell_phone",
    "television": "tv",
    "tv monitor": "tv",
}

VALID_WORD_SOURCES = {"visible_grounded_words", "selected_words"}
VALID_WORD_MATCH_MODES = {"exact", "exact_alias", "hybrid_safe"}
VALID_WORD_NO_MATCH_BEHAVIORS = {"keep_masks", "keep_best_similar", "filter_out"}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_word(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.lower().replace("_", " ").replace("-", " ")
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def _singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith("sses") and len(token) > 5:
        return token[:-2]
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalized_variants(text: str) -> List[str]:
    normalized = normalize_word(text)
    if not normalized:
        return []

    variants = [normalized]
    singular = " ".join(_singularize_token(token) for token in normalized.split()).strip()
    if singular and singular not in variants:
        variants.append(singular)

    compact = normalized.replace(" ", "")
    if compact and compact not in variants:
        variants.append(compact)

    singular_compact = singular.replace(" ", "")
    if singular_compact and singular_compact not in variants:
        variants.append(singular_compact)
    return variants


def canonicalize_word_list(values: Any) -> List[str]:
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = _SPACE_RE.sub(" ", value.strip())
        if not value:
            continue
        cleaned.append(value)
    return _dedupe_preserve_order(cleaned)


def lookup_class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_PANOPTIC_CLASS_NAMES):
        return COCO_PANOPTIC_CLASS_NAMES[class_id]
    return f"cls{class_id}"


def coco_class_id_from_name(class_name: str) -> Optional[int]:
    normalized_target = normalize_word(class_name)
    for class_id, candidate in enumerate(COCO_PANOPTIC_CLASS_NAMES):
        if normalize_word(candidate) == normalized_target:
            return class_id
    return None


@dataclass
class WordClassMatchConfig:
    enable: bool = True
    source: str = "visible_grounded_words"
    mode: str = "hybrid_safe"
    no_match_behavior: str = "keep_masks"
    similarity_threshold: float = 0.86

    def __post_init__(self) -> None:
        if self.source not in VALID_WORD_SOURCES:
            raise ValueError(
                f"Invalid word match source '{self.source}'. "
                f"Valid: {sorted(VALID_WORD_SOURCES)}"
            )
        if self.mode not in VALID_WORD_MATCH_MODES:
            raise ValueError(
                f"Invalid word match mode '{self.mode}'. "
                f"Valid: {sorted(VALID_WORD_MATCH_MODES)}"
            )
        if self.no_match_behavior not in VALID_WORD_NO_MATCH_BEHAVIORS:
            raise ValueError(
                f"Invalid word no-match behavior '{self.no_match_behavior}'. "
                f"Valid: {sorted(VALID_WORD_NO_MATCH_BEHAVIORS)}"
            )
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"mm_eomt_word_match_similarity_threshold must be in [0, 1], got {self.similarity_threshold}."
            )

    @classmethod
    def from_config(cls, config: Any) -> "WordClassMatchConfig":
        return cls(
            enable=bool(getattr(config, "mm_eomt_word_match_enable", True)),
            source=str(getattr(config, "mm_eomt_word_match_source", "visible_grounded_words")),
            mode=str(getattr(config, "mm_eomt_word_match_mode", "hybrid_safe")),
            no_match_behavior=str(getattr(config, "mm_eomt_word_match_no_match", "keep_masks")),
            similarity_threshold=float(getattr(config, "mm_eomt_word_match_similarity_threshold", 0.86)),
        )


@dataclass
class WordClassMatchResult:
    requested_source: str
    source_used: str
    source_note: Optional[str]
    mode: str
    no_match_behavior: str
    similarity_threshold: float
    input_words: List[str] = field(default_factory=list)
    candidate_class_names: List[str] = field(default_factory=list)
    matched_words: List[str] = field(default_factory=list)
    unmatched_words: List[str] = field(default_factory=list)
    matched_class_names: List[str] = field(default_factory=list)
    kept_class_names: List[str] = field(default_factory=list)
    word_matches: List[Dict[str, Any]] = field(default_factory=list)
    best_similar_word: Optional[str] = None
    best_similar_class_name: Optional[str] = None
    best_similarity: float = 0.0
    filter_applied: bool = False
    filter_reason: str = "no_words_available"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_source": self.requested_source,
            "source_used": self.source_used,
            "source_note": self.source_note,
            "mode": self.mode,
            "no_match_behavior": self.no_match_behavior,
            "similarity_threshold": self.similarity_threshold,
            "input_words": list(self.input_words),
            "candidate_class_names": list(self.candidate_class_names),
            "matched_words": list(self.matched_words),
            "unmatched_words": list(self.unmatched_words),
            "matched_class_names": list(self.matched_class_names),
            "kept_class_names": list(self.kept_class_names),
            "word_matches": list(self.word_matches),
            "best_similar_word": self.best_similar_word,
            "best_similar_class_name": self.best_similar_class_name,
            "best_similarity": self.best_similarity,
            "filter_applied": self.filter_applied,
            "filter_reason": self.filter_reason,
        }


def resolve_words_from_entry(
    entry: Optional[Dict[str, Any]],
    requested_source: str,
) -> Tuple[List[str], str, Optional[str]]:
    entry = entry if isinstance(entry, dict) else {}
    requested_words = canonicalize_word_list(entry.get(requested_source, []))
    if requested_words:
        return requested_words, requested_source, None

    fallback_source = "selected_words" if requested_source == "visible_grounded_words" else "visible_grounded_words"
    fallback_words = canonicalize_word_list(entry.get(fallback_source, []))
    if fallback_words:
        return fallback_words, fallback_source, f"fallback_to_{fallback_source}"

    return [], requested_source, "no_words_for_requested_source"


def _build_candidate_form_index(class_names: Sequence[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for class_name in class_names:
        if not isinstance(class_name, str):
            continue
        for variant in _normalized_variants(class_name):
            index.setdefault(variant, class_name)
    return index


def _match_word_exact_or_alias(
    word: str,
    *,
    mode: str,
    candidate_form_index: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], float]:
    for variant in _normalized_variants(word):
        matched_class = candidate_form_index.get(variant)
        if matched_class is not None:
            return matched_class, "exact", 1.0

    if mode == "exact":
        return None, None, 0.0

    for variant in _normalized_variants(word):
        alias_target = COCO_WORD_CLASS_ALIASES.get(variant)
        if alias_target is None:
            continue
        for target_variant in _normalized_variants(alias_target):
            matched_class = candidate_form_index.get(target_variant)
            if matched_class is not None:
                return matched_class, "alias", 1.0

    return None, None, 0.0


def _safe_similarity_score(word: str, class_name: str) -> float:
    word_norm = normalize_word(word)
    class_norm = normalize_word(class_name)
    if not word_norm or not class_norm:
        return 0.0

    if word_norm.replace(" ", "") == class_norm.replace(" ", ""):
        return 1.0

    word_tokens = word_norm.split()
    class_tokens = class_norm.split()
    shared_tokens = set(word_tokens) & set(class_tokens)
    if not shared_tokens:
        return 0.0

    overlap = len(shared_tokens) / max(len(set(word_tokens)), len(set(class_tokens)))
    compact_ratio = difflib.SequenceMatcher(
        None,
        word_norm.replace(" ", ""),
        class_norm.replace(" ", ""),
    ).ratio()
    if len(word_tokens) != len(class_tokens):
        compact_ratio *= 0.85

    length_ratio = min(len(word_norm), len(class_norm)) / max(len(word_norm), len(class_norm))
    return (0.55 * compact_ratio) + (0.30 * overlap) + (0.15 * length_ratio)


def _best_hybrid_candidate(
    word: str,
    candidate_class_names: Sequence[str],
) -> Tuple[Optional[str], float]:
    best_class: Optional[str] = None
    best_score = 0.0
    for candidate in candidate_class_names:
        score = _safe_similarity_score(word, candidate)
        if score > best_score:
            best_class = candidate
            best_score = score
    return best_class, best_score


def match_words_to_class_names(
    words: Sequence[str],
    *,
    candidate_class_names: Optional[Sequence[str]] = None,
    match_config: Optional[WordClassMatchConfig] = None,
    requested_source: str = "visible_grounded_words",
    source_used: Optional[str] = None,
    source_note: Optional[str] = None,
) -> WordClassMatchResult:
    match_config = match_config or WordClassMatchConfig()
    candidate_class_names = _dedupe_preserve_order(
        class_name
        for class_name in (candidate_class_names or COCO_PANOPTIC_CLASS_NAMES)
        if isinstance(class_name, str) and class_name
    )
    input_words = canonicalize_word_list(list(words))
    result = WordClassMatchResult(
        requested_source=requested_source,
        source_used=source_used or requested_source,
        source_note=source_note,
        mode=match_config.mode,
        no_match_behavior=match_config.no_match_behavior,
        similarity_threshold=match_config.similarity_threshold,
        input_words=input_words,
        candidate_class_names=list(candidate_class_names),
    )

    if not match_config.enable:
        result.filter_reason = "word_match_disabled"
        return result
    if not input_words:
        result.filter_reason = "no_words_available"
        return result
    if not candidate_class_names:
        result.filter_reason = "no_candidate_classes"
        return result

    candidate_form_index = _build_candidate_form_index(candidate_class_names)
    matched_class_names: List[str] = []
    matched_words: List[str] = []
    unmatched_words: List[str] = []
    word_matches: List[Dict[str, Any]] = []
    best_similar_word: Optional[str] = None
    best_similar_class_name: Optional[str] = None
    best_similarity = 0.0

    for word in input_words:
        matched_class, method, score = _match_word_exact_or_alias(
            word,
            mode=match_config.mode,
            candidate_form_index=candidate_form_index,
        )

        if matched_class is None and match_config.mode == "hybrid_safe":
            hybrid_class, hybrid_score = _best_hybrid_candidate(word, candidate_class_names)
            if hybrid_class is not None and hybrid_score >= match_config.similarity_threshold:
                matched_class = hybrid_class
                method = "hybrid_safe"
                score = hybrid_score
            elif hybrid_class is not None and hybrid_score > best_similarity:
                best_similar_word = word
                best_similar_class_name = hybrid_class
                best_similarity = hybrid_score
        else:
            hybrid_class, hybrid_score = _best_hybrid_candidate(word, candidate_class_names)
            if hybrid_class is not None and hybrid_score > best_similarity:
                best_similar_word = word
                best_similar_class_name = hybrid_class
                best_similarity = hybrid_score

        if matched_class is not None and method is not None:
            matched_words.append(word)
            matched_class_names.append(matched_class)
            word_matches.append(
                {
                    "word": word,
                    "class_name": matched_class,
                    "method": method,
                    "score": float(score),
                }
            )
        else:
            unmatched_words.append(word)

    result.matched_words = _dedupe_preserve_order(matched_words)
    result.unmatched_words = _dedupe_preserve_order(unmatched_words)
    result.matched_class_names = _dedupe_preserve_order(matched_class_names)
    result.word_matches = word_matches
    result.best_similar_word = best_similar_word
    result.best_similar_class_name = best_similar_class_name
    result.best_similarity = float(best_similarity)

    if result.matched_class_names:
        result.filter_applied = True
        result.kept_class_names = list(result.matched_class_names)
        result.filter_reason = "matched_classes"
        return result

    if match_config.no_match_behavior == "keep_masks":
        result.filter_reason = "no_word_class_match_keep_masks"
        return result

    if match_config.no_match_behavior == "filter_out":
        result.filter_applied = True
        result.kept_class_names = []
        result.filter_reason = "no_word_class_match_filter_out"
        return result

    recovery_floor = max(0.50, match_config.similarity_threshold * 0.75)
    if best_similar_class_name is not None and best_similarity >= recovery_floor:
        result.filter_applied = True
        result.kept_class_names = [best_similar_class_name]
        result.filter_reason = "no_word_class_match_keep_best_similar"
        return result

    result.filter_reason = "no_word_class_match_keep_masks"
    return result


def match_entry_words_to_class_names(
    entry: Optional[Dict[str, Any]],
    candidate_class_names: Optional[Sequence[str]] = None,
    match_config: Optional[WordClassMatchConfig] = None,
) -> WordClassMatchResult:
    match_config = match_config or WordClassMatchConfig()
    words, source_used, source_note = resolve_words_from_entry(
        entry,
        requested_source=match_config.source,
    )
    return match_words_to_class_names(
        words,
        candidate_class_names=candidate_class_names,
        match_config=match_config,
        requested_source=match_config.source,
        source_used=source_used,
        source_note=source_note,
    )
