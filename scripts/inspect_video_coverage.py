"""
inspect_video_coverage.py

Checks which video files on disk are referenced by the training JSON files
(vsibench_train + vstibench_train), and which JSON-referenced videos are
missing from disk.

Run on HPC:
    python $REPO/scripts/inspect_video_coverage.py \
        --data-root /leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r \
        --qa-root   /leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/VLM-3R-DATA \
        --output    $REPO/logs/video_coverage_report.txt
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_video_files(video_dir: Path) -> set:
    """Return a set of stem names (no extension) for all video files under video_dir."""
    stems = set()
    if not video_dir.exists():
        return stems
    for p in video_dir.rglob("*"):
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            stems.add(p.stem)
    return stems


def extract_video_stems_from_jsons(json_paths: list) -> dict:
    """
    Returns {json_filename: set_of_stems} for each JSON file.
    Uses the `video` field; falls back to `scene_name` if present.
    """
    result = {}
    for jp in json_paths:
        stems = set()
        data = json.load(open(jp))
        for item in data:
            vid = item.get("video")
            if vid:
                stems.add(Path(vid).stem)
            elif "scene_name" in item:
                stems.add(item["scene_name"])
        result[Path(jp).name] = stems
    return result


def report_section(title: str, lines: list) -> list:
    out = []
    out.append("")
    out.append("=" * 70)
    out.append(title)
    out.append("=" * 70)
    out.extend(lines)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Video ↔ JSON coverage checker")
    parser.add_argument("--data-root", required=True,
                        help="Root dir of video datasets "
                             "(e.g. /leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r)")
    parser.add_argument("--qa-root", required=True,
                        help="Root dir containing VLM-3R-DATA/ "
                             "(e.g. /leonardo_scratch/.../VLM-3R-DATA)")
    parser.add_argument("--output", default=None,
                        help="Optional path to write the report to (also prints to stdout)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    qa_root   = Path(args.qa_root)

    # ------------------------------------------------------------------
    # 1. Collect video files on disk, per dataset
    # ------------------------------------------------------------------
    # vsibench / pi3x extraction uses these directories:
    disk_video_dirs = {
        "scannet":     data_root / "scannet"     / "videos",
        "scannetpp":   data_root / "scannetpp"   / "videos",
        "arkitscenes": data_root / "arkitscenes" / "videos",
    }
    # vstibench uses a separate ScanNet directory
    vstibench_scannet_dir = data_root / "ScanNet" / "videos" / "train"

    disk_stems = {k: scan_video_files(v) for k, v in disk_video_dirs.items()}
    disk_stems["ScanNet_train"] = scan_video_files(vstibench_scannet_dir)

    # ------------------------------------------------------------------
    # 2. Collect JSON files
    # ------------------------------------------------------------------
    vsibench_dir  = qa_root / "vsibench_train"
    vstibench_dir = qa_root / "vstibench_train"

    vsibench_jsons  = sorted(vsibench_dir.glob("*.json"))
    vstibench_jsons = sorted(vstibench_dir.glob("*.json"))

    all_jsons = vsibench_jsons + vstibench_jsons

    if not all_jsons:
        print(f"ERROR: No JSON files found under {qa_root}")
        return

    stems_by_json = extract_video_stems_from_jsons([str(j) for j in all_jsons])

    # ------------------------------------------------------------------
    # 3. Aggregate: which stems appear in ANY json (vsibench / vstibench)
    # ------------------------------------------------------------------
    vsibench_all_stems  = set()
    vstibench_all_stems = set()
    for j in vsibench_jsons:
        vsibench_all_stems  |= stems_by_json[j.name]
    for j in vstibench_jsons:
        vstibench_all_stems |= stems_by_json[j.name]

    all_json_stems = vsibench_all_stems | vstibench_all_stems

    # ------------------------------------------------------------------
    # 4. Build report
    # ------------------------------------------------------------------
    lines = []
    lines += ["VIDEO COVERAGE REPORT", ""]
    lines += [f"data_root : {data_root}"]
    lines += [f"qa_root   : {qa_root}"]
    lines += [""]

    # --- Disk summary ---
    lines += report_section("DISK: Video file counts", [
        f"  {'Dataset':<22} {'Directory':<55} {'Video files'}",
        f"  {'-'*22} {'-'*55} {'-'*11}",
    ])
    for key, d in disk_video_dirs.items():
        exists = "✓" if d.exists() else "✗ (not found)"
        lines.append(f"  {key:<22} {str(d):<55} {len(disk_stems[key]):>6}  {exists}")
    lines.append(f"  {'ScanNet_train (vstibench)':<22} {str(vstibench_scannet_dir):<55} {len(disk_stems['ScanNet_train']):>6}"
                 + ("  ✓" if vstibench_scannet_dir.exists() else "  ✗ (not found)"))

    # --- Per-JSON summary ---
    lines += report_section("JSON: Unique videos referenced per file", [
        f"  {'JSON file':<55} {'Unique scenes'}",
        f"  {'-'*55} {'-'*13}",
    ])
    for jname, stems in sorted(stems_by_json.items()):
        lines.append(f"  {jname:<55} {len(stems):>6}")

    lines.append("")
    lines.append(f"  vsibench total unique scenes : {len(vsibench_all_stems)}")
    lines.append(f"  vstibench total unique scenes: {len(vstibench_all_stems)}")
    lines.append(f"  Combined unique scenes       : {len(all_json_stems)}")

    # --- Coverage: vsibench datasets vs disk ---
    lines += report_section("COVERAGE: vsibench JSONs vs disk (scannet / scannetpp / arkitscenes)", [])

    # vsibench uses scannet, scannetpp, arkitscenes directories
    # We determine which dataset a stem belongs to by checking which disk dir it appears in
    vsi_stems_per_dataset = defaultdict(set)
    for jpath in vsibench_jsons:
        data = json.load(open(str(jpath)))
        for item in data:
            src = item.get("data_source", "unknown")
            vid = item.get("video")
            if vid:
                vsi_stems_per_dataset[src].add(Path(vid).stem)
            elif "scene_name" in item:
                vsi_stems_per_dataset[src].add(item["scene_name"])

    for src, json_stems in sorted(vsi_stems_per_dataset.items()):
        disk_key = src  # e.g. "scannet", "scannetpp", "arkitscenes"
        on_disk  = disk_stems.get(disk_key, set())
        in_json_on_disk    = json_stems & on_disk
        in_json_not_disk   = json_stems - on_disk
        on_disk_not_json   = on_disk - json_stems

        lines.append(f"\n  [{src}]  (vsibench)")
        lines.append(f"    JSON unique scenes   : {len(json_stems)}")
        lines.append(f"    Disk video files     : {len(on_disk)}")
        lines.append(f"    In JSON + on disk    : {len(in_json_on_disk)}")
        lines.append(f"    In JSON, MISSING disk: {len(in_json_not_disk)}")
        lines.append(f"    On disk, NOT in JSON : {len(on_disk_not_json)}")
        if on_disk:
            lines.append(f"    JSON coverage of disk: {100*len(in_json_on_disk)/len(on_disk):.1f}%")
        if in_json_not_disk:
            lines.append(f"    Missing from disk (first 20): "
                         + ", ".join(sorted(in_json_not_disk)[:20]))
        if on_disk_not_json:
            lines.append(f"    On disk, unused (first 20) : "
                         + ", ".join(sorted(on_disk_not_json)[:20]))

    # --- Coverage: vstibench vs ScanNet_train disk ---
    lines += report_section("COVERAGE: vstibench JSONs vs disk (ScanNet/videos/train)", [])

    vsti_stems_per_json = {}
    vsti_all = set()
    for jpath in vstibench_jsons:
        data = json.load(open(str(jpath)))
        s = set()
        for item in data:
            vid = item.get("video")
            stem = Path(vid).stem if vid else item.get("scene_name")
            if stem:
                s.add(stem)
        vsti_stems_per_json[jpath.name] = s
        vsti_all |= s

    on_disk_vsti  = disk_stems["ScanNet_train"]
    in_both       = vsti_all & on_disk_vsti
    missing_disk  = vsti_all - on_disk_vsti
    unused_disk   = on_disk_vsti - vsti_all

    lines.append(f"\n  [ScanNet_train] (vstibench all 11 JSONs combined)")
    lines.append(f"    JSON unique scenes   : {len(vsti_all)}")
    lines.append(f"    Disk video files     : {len(on_disk_vsti)}")
    lines.append(f"    In JSON + on disk    : {len(in_both)}")
    lines.append(f"    In JSON, MISSING disk: {len(missing_disk)}")
    lines.append(f"    On disk, NOT in JSON : {len(unused_disk)}")
    if on_disk_vsti:
        lines.append(f"    JSON coverage of disk: {100*len(in_both)/len(on_disk_vsti):.1f}%")

    lines.append(f"\n  Per-JSON breakdown:")
    lines.append(f"    {'JSON file':<55} {'Unique'} {'On disk'} {'Missing'}")
    lines.append(f"    {'-'*55} {'-'*6} {'-'*7} {'-'*7}")
    for jname, s in sorted(vsti_stems_per_json.items()):
        on_d   = len(s & on_disk_vsti)
        miss   = len(s - on_disk_vsti)
        lines.append(f"    {jname:<55} {len(s):>6} {on_d:>7} {miss:>7}")

    if missing_disk:
        lines.append(f"\n  vstibench scenes missing from disk (first 30):")
        for sc in sorted(missing_disk)[:30]:
            lines.append(f"    {sc}")
    if unused_disk:
        lines.append(f"\n  ScanNet_train videos not used in any vstibench JSON (first 30):")
        for sc in sorted(unused_disk)[:30]:
            lines.append(f"    {sc}")

    # --- Final cross-check: spatial features coverage ---
    lines += report_section("SPATIAL FEATURES: .pt file coverage", [])

    feature_dirs = {
        "scannet (cut3r)":     data_root / "scannet"     / "spatial_features",
        "scannetpp (cut3r)":   data_root / "scannetpp"   / "spatial_features",
        "arkitscenes (cut3r)": data_root / "arkitscenes" / "spatial_features",
        "scannet (pi3x)":      data_root / "scannet"     / "spatial_features_pi3x",
        "scannetpp (pi3x)":    data_root / "scannetpp"   / "spatial_features_pi3x",
        "arkitscenes (pi3x)":  data_root / "arkitscenes" / "spatial_features_pi3x",
    }
    dataset_map = {
        "scannet (cut3r)":     "scannet",
        "scannetpp (cut3r)":   "scannetpp",
        "arkitscenes (cut3r)": "arkitscenes",
        "scannet (pi3x)":      "scannet",
        "scannetpp (pi3x)":    "scannetpp",
        "arkitscenes (pi3x)":  "arkitscenes",
    }

    lines.append(f"  {'Label':<26} {'Dir exists'} {'PT files'} {'All JSON stems covered'}")
    lines.append(f"  {'-'*26} {'-'*10} {'-'*8} {'-'*22}")
    for label, feat_dir in feature_dirs.items():
        exists = feat_dir.exists()
        if exists:
            pt_stems = {p.stem for p in feat_dir.rglob("*.pt")}
            n_pt = len(pt_stems)
            # Compare against the relevant vsibench JSON stems for that dataset
            ds_key = dataset_map[label]
            ref_stems = vsi_stems_per_dataset.get(ds_key, set())
            covered = ref_stems & pt_stems
            missing = ref_stems - pt_stems
            cov_str = f"{len(covered)}/{len(ref_stems)} ({100*len(covered)/max(len(ref_stems),1):.0f}%)"
        else:
            n_pt = 0
            cov_str = "N/A"
        lines.append(f"  {label:<26} {'yes' if exists else 'no':<10} {n_pt:>8} {cov_str}")

    # ------------------------------------------------------------------
    # 5. Print and/or write
    # ------------------------------------------------------------------
    report_text = "\n".join(lines)
    print(report_text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text)
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
