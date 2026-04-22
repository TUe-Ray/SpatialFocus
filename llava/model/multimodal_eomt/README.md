# `multimodal_eomt` Folder Guide

This folder contains the EoMT-based side branch used by VLM-3R for mask extraction, object-token selection, and selective 3D gating.

The short version:

1. `eomt_extractor.py` runs the frozen EoMT segmentation model and returns soft masks, class logits, and frame metadata.
2. `mask_pooler.py` converts dense visual grid features into object-centric pooled tokens using the selected masks.
3. `eomt_selector.py` picks which pooled object tokens should survive for the object block.
4. `eomt_object_block.py` inserts those object tokens before or after the visual tokens.
5. `eomt_obj_info.py` optionally adds an `OBJ_INFO`-style prefix token to the object block.
6. `selective_3d_gate.py` uses EoMT masks to gate only the 3D spatial tokens.
7. `word_class_matcher.py` matches question-selected words to COCO class names so word-aware selection can safely keep or suppress masks.

## File-by-file

### `__init__.py`
- Re-exports the public classes and functions from this folder.
- If another module imports `llava.model.multimodal_eomt`, this is the main entrypoint.

### `eomt_extractor.py`
- Wraps the EoMT model from `third_party/EoMT`.
- Loads the YAML config and checkpoint.
- Runs inference on frames and returns:
  - `soft_masks`
  - `class_logits`
  - `frame_meta`
  - `stuff_class_ids`
- This is the earliest place where raw segmentation outputs enter the VLM-3R pipeline.

### `mask_pooler.py`
- Takes EoMT masks plus visual features and pools object-centric tokens.
- Supports several query-selection heuristics such as class confidence and mask confidence.
- Produces per-frame selected query indices, scores, and class IDs.
- This is the main bridge from dense masks to compact object tokens.

### `eomt_selector.py`
- Takes the pooled object tokens and performs the final object-token selection for the object block.
- Supports:
  - `class_aware`
  - `external_socket`
- `external_socket` is the path that consumes externally provided:
  - `selected_query_ids`
  - `selected_mask_ids`
  - `visible_grounded_words`
  - `selected_words`
- After the recent patch, words-only entries are no longer deferred; they are matched against candidate class names using `word_class_matcher.py`.

### `eomt_object_block.py`
- Very small utility that appends or prepends selected object tokens around the visual tokens.
- It does not decide which objects to keep; it only merges tensors.

### `eomt_obj_info.py`
- Builds an optional prefix token block for the object tokens.
- Supports:
  - `none`
  - `text_phrase`
  - `learnable_embedding`
- This is useful when the object block should carry a lightweight “object info” marker.

### `selective_3d_gate.py`
- Applies EoMT-based gating to the 3D patch tokens only.
- Main steps:
  1. compute per-query foreground confidence
  2. optionally filter by thing/stuff
  3. optionally filter by question words via `word_class_matcher.py`
  4. threshold and top-k select queries
  5. merge the selected masks into one gate
  6. apply the gate to the 3D features
- Debug output is stored in `SelectiveGateDebugInfo`.
- This is the file to touch if the goal is question-aware mask filtering for selective 3D fusion.

### `word_class_matcher.py`
- Shared utility for mapping selected question words to COCO class names.
- Centralizes:
  - normalization
  - simple singularization
  - curated alias mapping like `sofa -> couch`
  - matching modes:
    - `exact`
    - `exact_alias`
    - `hybrid_safe`
  - no-match policies:
    - `keep_masks`
    - `keep_best_similar`
    - `filter_out`
- This file is used by both:
  - `eomt_selector.py`
  - `selective_3d_gate.py`
- If future work changes the taxonomy, matching rules, or alias coverage, start here.

## Main runtime flows

### Flow A: Object block path
1. `eomt_extractor.py` produces masks and class logits.
2. `mask_pooler.py` converts them into pooled object tokens.
3. `eomt_selector.py` keeps a subset of those object tokens.
4. `eomt_obj_info.py` optionally builds prefix tokens.
5. `eomt_object_block.py` inserts the object tokens into the final visual sequence.

### Flow B: Selective 3D path
1. `eomt_extractor.py` produces masks and class logits.
2. `selective_3d_gate.py` scores and filters mask queries.
3. A merged gate is built from the selected masks.
4. The gate is applied to 3D patch tokens only.

## Where question words enter

Question-side words are expected to arrive in `frame_meta` or the external selection socket.

Relevant fields:
- `visible_grounded_words`
- `selected_words`

Current default behavior:
- use `visible_grounded_words`
- match with `hybrid_safe`
- if there is no reliable class match, keep masks instead of dropping them

This is intentional because COCO class coverage is limited for indoor categories such as `cabinet`, `desk`, `whiteboard`, and `radiator`.

## Config knobs for selection and word matching

Object-block selection keeps the original post-pool namespace:

- `mm_eomt_selector_mode`
- `mm_eomt_selector_keep_stuff`
- `mm_eomt_selector_keep_things`
- `mm_eomt_selector_drop_no_object`
- `mm_eomt_selector_order`
- `mm_eomt_selector_no_object_class_id`
- `mm_eomt_word_match_enable`
- `mm_eomt_word_match_source`
- `mm_eomt_word_match_mode`
- `mm_eomt_word_match_no_match`
- `mm_eomt_word_match_similarity_threshold`

Selective 3D uses a separate namespace so it can coexist with the object block:

- `mm_eomt_selective_3d_selector_mode`
- `mm_eomt_selective_3d_score_threshold`
- `mm_eomt_selective_3d_topk`
- `mm_eomt_selective_3d_class_type`
- `mm_eomt_selective_3d_word_match_enable`
- `mm_eomt_selective_3d_word_match_source`
- `mm_eomt_selective_3d_word_match_mode`
- `mm_eomt_selective_3d_word_match_no_match`
- `mm_eomt_selective_3d_word_match_similarity_threshold`
- `mm_eomt_selective_3d_merge_mode`
- `mm_eomt_selective_3d_gate_type`
- `mm_eomt_selective_3d_floor`
- `mm_eomt_selective_3d_empty_fallback`

Experiment families now live in:

- `configs/eomt/eomt_objinfo_round1.json`
- `configs/eomt/eomt_selective_3d_round1.json`
- `configs/eomt/eomt_combined_round1.json`

## If another agent needs to modify behavior

### If the task is “improve word/class matching”
- Start with `word_class_matcher.py`
- Then inspect:
  - `eomt_selector.py`
  - `selective_3d_gate.py`

### If the task is “change which objects become object-block tokens”
- Start with:
  - `mask_pooler.py`
  - `eomt_selector.py`

### If the task is “change the 3D gating behavior”
- Start with `selective_3d_gate.py`

### If the task is “change how EoMT outputs are produced”
- Start with `eomt_extractor.py`

## Practical caveats

- This folder assumes COCO-like class IDs for readable class names and aliases.
- `frame_meta` is important. If metadata does not propagate, question-aware matching will silently become less useful.
- Some repo-level imports pull optional dependencies like `transformers` and video packages. Lightweight validation scripts should avoid depending on the full LM stack when possible.
- `scripts/validate_eomt_experiment_configs.py` validates the experiment-family resolver and the legacy selective-3D key aliasing without requiring Torch.
