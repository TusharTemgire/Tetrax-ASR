"""
AI4Bharat IndicVoices Dataset Inspection and Download CLI (Marathi & Hindi)
Inspects IndicVoices subsets, checks local files in data/, and optionally saves splits to disk.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tetrax_asr.data.indicvoices_loader import (
    load_raw_indicvoices,
    find_local_indicvoices_files,
    prepare_indicvoices_dataset_dict
)


def main():
    parser = argparse.ArgumentParser(description="Inspect and prepare AI4Bharat IndicVoices dataset (Marathi & Hindi)")
    parser.add_argument("--languages", nargs="+", default=["mr", "hi"], help="Languages to load (mr, hi)")
    parser.add_argument("--save_dir", type=str, default="", help="Optional directory to save splits")
    args = parser.parse_args()

    print("=" * 65)
    print("   AI4Bharat IndicVoices (Marathi + Hindi) Explorer (Tetrax ASR)")
    print("=" * 65)

    print("\nChecking for local dataset files in data/ directory...")
    local_files = find_local_indicvoices_files()
    for lang, files in local_files.items():
        print(f"  • {lang.upper()}: {len(files)} parquet file(s) found locally")

    print(f"\nLoading IndicVoices for languages: {args.languages}...")
    try:
        dataset = load_raw_indicvoices(languages=args.languages)
        print(f"Total merged samples: {len(dataset)}")

        # Count samples per language
        lang_counts = {}
        for sample in dataset:
            l = sample.get("language", "unknown")
            lang_counts[l] = lang_counts.get(l, 0) + 1

        print("\nSamples per language:")
        for l, count in lang_counts.items():
            print(f"  • {l}: {count} samples")

        if args.save_dir:
            print(f"\nSaving stratified train/val/test splits to {args.save_dir}...")
            prepare_indicvoices_dataset_dict(languages=args.languages, save_dir=args.save_dir)
            print("Splits successfully saved to disk.")

    except Exception as e:
        print(f"\nNotice: {e}")
        print("Tip: If running without internet or HF authentication, place parquet files into data/marathi and data/hindi.")

    print("\nDone.")


if __name__ == "__main__":
    main()
