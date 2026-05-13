#!/usr/bin/env python3

# Copyright 2026 ESPnet Contributors
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Mixed Error Rate (MER) evaluation for Trilingual Code-Switched ASR.

This script evaluates Mixed Error Rate for code-switched speech recognition
with support for three languages: Arabic (AR), English (EN), and a third language (ID).

MER measures errors in code-switching regions separately from intra-language regions,
providing insights into:
- Overall error rate
- Per-language error rates (AR, EN, ID)
- Code-switching error rates (errors at language switches)
- Monolingual vs code-switching performance
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import editdistance
import numpy as np


# Language identifiers
SUPPORTED_LANGUAGES = {"AR", "EN", "ID"}
LANG_DELIM = "<"  # e.g., <AR>, <EN>, <ID>


class MERCalculator:
    """Calculate Mixed Error Rate metrics for code-switched ASR."""

    def __init__(
        self,
        languages: List[str] = ["AR", "EN", "ID"],
        lang_delim: str = "<",
        report_per_lang: bool = True,
        report_cs_regions: bool = True,
    ):
        """Initialize MER Calculator.

        Args:
            languages: List of language codes (e.g., ["AR", "EN", "ID"])
            lang_delim: Language tag delimiter (default: "<" for <AR>, <EN>, etc.)
            report_per_lang: Report per-language metrics
            report_cs_regions: Report code-switching region metrics
        """
        self.languages = languages
        self.lang_delim = lang_delim
        self.report_per_lang = report_per_lang
        self.report_cs_regions = report_cs_regions

        # Initialize metrics storage
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            "overall": {"errors": 0, "total": 0, "wer": 0.0, "cer": 0.0},
            "per_language": {
                lang: {"errors": 0, "total": 0, "wer": 0.0, "cer": 0.0}
                for lang in self.languages
            },
            "code_switching": {"errors": 0, "total": 0, "wer": 0.0, "cer": 0.0},
            "monolingual": {"errors": 0, "total": 0, "wer": 0.0, "cer": 0.0},
        }

    def extract_language_tags(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract language tags and their positions from text.

        Args:
            text: Text with embedded language tags (e.g., "hello <EN> world <AR> مرحبا")

        Returns:
            List of tuples: (start_pos, end_pos, language_code)
        """
        pattern = rf"{self.lang_delim}({self.lang_delim}|".join(self.languages) + rf"){self.lang_delim}"
        tags = []
        for match in re.finditer(pattern, text):
            start, end = match.span()
            lang = match.group(1)
            tags.append((start, end, lang))
        return tags

    def segment_by_language(
        self, text: str
    ) -> List[Tuple[str, str, Tuple[int, int]]]:
        """Segment text by language and return language-annotated segments.

        Args:
            text: Text with language tags

        Returns:
            List of tuples: (segment_text, language, (start, end))
        """
        tags = self.extract_language_tags(text)
        if not tags:
            # No language tags found
            return [(text, "UNKNOWN", (0, len(text)))]

        segments = []
        current_lang = "UNKNOWN"
        current_pos = 0

        for start, end, lang in tags:
            # Add text before this tag
            if start > current_pos:
                segment_text = text[current_pos:start]
                if segment_text.strip():
                    segments.append((segment_text, current_lang, (current_pos, start)))

            current_lang = lang
            current_pos = end

        # Add remaining text
        if current_pos < len(text):
            segment_text = text[current_pos:]
            if segment_text.strip():
                segments.append((segment_text, current_lang, (current_pos, len(text))))

        return segments

    def detect_code_switches(
        self, segments: List[Tuple[str, str, Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """Detect code-switching regions (language changes).

        Args:
            segments: Segmented text with language labels

        Returns:
            List of (start, end) positions of code-switching regions
        """
        cs_regions = []
        if len(segments) < 2:
            return cs_regions

        for i in range(len(segments) - 1):
            _, lang1, (_, end1) = segments[i]
            _, lang2, (start2, _) = segments[i + 1]

            if lang1 != lang2 and lang1 != "UNKNOWN" and lang2 != "UNKNOWN":
                # Code-switch region between tokens around switch point
                # Typically a small window around the switch
                window = 5  # characters
                cs_start = max(0, end1 - window)
                cs_end = min(start2 + window, len(segments[i][0]) + len(segments[i + 1][0]))
                cs_regions.append((cs_start, cs_end))

        return cs_regions

    def is_in_cs_region(self, pos: int, cs_regions: List[Tuple[int, int]]) -> bool:
        """Check if position is in code-switching region.

        Args:
            pos: Position in text
            cs_regions: List of code-switching regions

        Returns:
            True if position is in CS region
        """
        for start, end in cs_regions:
            if start <= pos <= end:
                return True
        return False

    def calculate_wer(
        self, hyp_words: List[str], ref_words: List[str]
    ) -> Tuple[float, List[str]]:
        """Calculate Word Error Rate.

        Args:
            hyp_words: Hypothesis words
            ref_words: Reference words

        Returns:
            (WER, error_types_per_word)
        """
        distance = editdistance.eval(hyp_words, ref_words)
        wer = float(distance) / len(ref_words) if len(ref_words) > 0 else 0.0
        return wer, distance

    def calculate_cer(
        self, hyp_text: str, ref_text: str
    ) -> Tuple[float, int]:
        """Calculate Character Error Rate.

        Args:
            hyp_text: Hypothesis text
            ref_text: Reference text

        Returns:
            (CER, num_errors)
        """
        # Remove spaces and language tags for CER
        hyp_clean = re.sub(rf"{self.lang_delim}[A-Z]{2}{self.lang_delim}", "", hyp_text)
        ref_clean = re.sub(rf"{self.lang_delim}[A-Z]{2}{self.lang_delim}", "", ref_text)
        hyp_clean = hyp_clean.replace(" ", "")
        ref_clean = ref_clean.replace(" ", "")

        distance = editdistance.eval(hyp_clean, ref_clean)
        cer = float(distance) / len(ref_clean) if len(ref_clean) > 0 else 0.0
        return cer, distance

    def process_pair(
        self, hypothesis: str, reference: str
    ) -> Dict[str, Dict[str, float]]:
        """Process a hypothesis-reference pair and calculate metrics.

        Args:
            hypothesis: Model output (may contain language tags)
            reference: Ground truth (should contain language tags for accurate evaluation)

        Returns:
            Dictionary of metrics for this utterance
        """
        results = {
            "overall": {},
            "per_language": {},
            "code_switching": {},
            "monolingual": {},
        }

        # Segment both hypothesis and reference by language
        ref_segments = self.segment_by_language(reference)
        hyp_text_clean = re.sub(
            rf"{self.lang_delim}[A-Z]{2}{self.lang_delim}", "", hypothesis
        )

        # Detect code-switching regions in reference
        cs_regions = self.detect_code_switches(ref_segments)

        # Calculate overall metrics
        ref_words = reference.split()
        hyp_words = hyp_text_clean.split()

        overall_wer, overall_wer_count = self.calculate_wer(hyp_words, ref_words)
        overall_cer, overall_cer_count = self.calculate_cer(hypothesis, reference)

        results["overall"]["wer"] = overall_wer
        results["overall"]["cer"] = overall_cer
        results["overall"]["word_errors"] = overall_wer_count
        results["overall"]["char_errors"] = overall_cer_count
        results["overall"]["total_words"] = len(ref_words)
        results["overall"]["total_chars"] = len(
            ref.replace(" ", "")
            for _, ref, _ in ref_segments
        )

        # Calculate per-language metrics
        if self.report_per_lang:
            for lang in self.languages:
                lang_ref_text = " ".join(
                    [seg for seg, seg_lang, _ in ref_segments if seg_lang == lang]
                )
                lang_ref_words = lang_ref_text.split()

                # Find corresponding hypothesis for this language
                lang_hyp_text = " ".join(
                    [seg for seg, seg_lang, _ in ref_segments if seg_lang == lang]
                )
                lang_hyp_words = lang_hyp_text.split()

                if len(lang_ref_words) > 0:
                    lang_wer, lang_wer_count = self.calculate_wer(
                        lang_hyp_words, lang_ref_words
                    )
                    lang_cer, lang_cer_count = self.calculate_cer(
                        lang_hyp_text, lang_ref_text
                    )
                    results["per_language"][lang] = {
                        "wer": lang_wer,
                        "cer": lang_cer,
                        "word_errors": lang_wer_count,
                        "char_errors": lang_cer_count,
                        "total_words": len(lang_ref_words),
                    }

        # Calculate code-switching vs monolingual metrics
        if self.report_cs_regions and cs_regions:
            # This would require character-level alignment
            # Simplified version: flag if utterance contains code-switches
            results["code_switching"]["has_cs"] = True
            results["code_switching"]["wer"] = overall_wer
            results["code_switching"]["cer"] = overall_cer
        elif self.report_cs_regions:
            results["monolingual"]["wer"] = overall_wer
            results["monolingual"]["cer"] = overall_cer

        return results

    def update_metrics(self, results: Dict[str, Dict[str, float]]):
        """Update cumulative metrics with results from one utterance.

        Args:
            results: Metrics from process_pair
        """
        # Update overall
        if "total_words" in results["overall"]:
            total_words = results["overall"]["total_words"]
            word_errors = results["overall"]["word_errors"]
            self.metrics["overall"]["errors"] += word_errors
            self.metrics["overall"]["total"] += total_words

        # Update per-language
        for lang in self.languages:
            if lang in results["per_language"]:
                lang_result = results["per_language"][lang]
                self.metrics["per_language"][lang]["errors"] += lang_result.get(
                    "word_errors", 0
                )
                self.metrics["per_language"][lang]["total"] += lang_result.get(
                    "total_words", 0
                )

    def get_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate metrics from accumulated data.

        Returns:
            Dictionary of final metrics
        """
        final_metrics = {}

        # Overall MER
        if self.metrics["overall"]["total"] > 0:
            final_metrics["overall_wer"] = (
                100.0
                * self.metrics["overall"]["errors"]
                / self.metrics["overall"]["total"]
            )
        else:
            final_metrics["overall_wer"] = 0.0

        # Per-language MER
        for lang in self.languages:
            if self.metrics["per_language"][lang]["total"] > 0:
                final_metrics[f"wer_{lang}"] = (
                    100.0
                    * self.metrics["per_language"][lang]["errors"]
                    / self.metrics["per_language"][lang]["total"]
                )
            else:
                final_metrics[f"wer_{lang}"] = 0.0

        return final_metrics


def load_text_file(filepath: str) -> Dict[str, str]:
    """Load text file in Kaldi format (key value).

    Args:
        filepath: Path to text file

    Returns:
        Dictionary mapping utterance IDs to text
    """
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                utt_id, text = parts
                data[utt_id] = text
    return data


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate Mixed Error Rate (MER) for code-switched ASR"
    )
    parser.add_argument(
        "ref_file",
        type=str,
        help="Path to reference text file (Kaldi format with language tags)",
    )
    parser.add_argument(
        "hyp_file",
        type=str,
        help="Path to hypothesis text file (Kaldi format)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for results",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="AR,EN,ID",
        help="Comma-separated list of language codes",
    )
    parser.add_argument(
        "--lang_delim",
        type=str,
        default="<",
        help="Language tag delimiter (e.g., '<' for '<AR>')",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=WARN, 1=INFO, 2=DEBUG)",
    )
    return parser


def main():
    """Main function."""
    args = get_parser().parse_args()

    # Setup logging
    if args.verbose == 0:
        level = logging.WARN
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Parse languages
    languages = [lang.strip().upper() for lang in args.languages.split(",")]
    logging.info(f"Languages: {languages}")

    # Load files
    logging.info(f"Loading reference: {args.ref_file}")
    ref_data = load_text_file(args.ref_file)

    logging.info(f"Loading hypothesis: {args.hyp_file}")
    hyp_data = load_text_file(args.hyp_file)

    # Initialize calculator
    calculator = MERCalculator(
        languages=languages,
        lang_delim=args.lang_delim,
        report_per_lang=True,
        report_cs_regions=True,
    )

    # Process all utterances
    logging.info("Processing utterances...")
    utt_results = {}

    for utt_id in ref_data.keys():
        if utt_id not in hyp_data:
            logging.warning(f"Missing hypothesis for {utt_id}")
            continue

        ref = ref_data[utt_id]
        hyp = hyp_data[utt_id]

        results = calculator.process_pair(hyp, ref)
        utt_results[utt_id] = results
        calculator.update_metrics(results)

    # Get final metrics
    final_metrics = calculator.get_aggregate_metrics()

    # Log results
    logging.info("=" * 50)
    logging.info("MIXED ERROR RATE (MER) Results")
    logging.info("=" * 50)
    logging.info(f"Overall WER: {final_metrics['overall_wer']:.2f}%")

    for lang in languages:
        if f"wer_{lang}" in final_metrics:
            logging.info(f"WER {lang}: {final_metrics[f'wer_{lang}']:.2f}%")

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    results_file = os.path.join(args.outdir, "mer_results.txt")
    with open(results_file, "w") as f:
        f.write("Mixed Error Rate (MER) Results\n")
        f.write("=" * 50 + "\n")
        for metric_name, value in final_metrics.items():
            f.write(f"{metric_name}: {value:.2f}%\n")

    logging.info(f"Results saved to {results_file}")

    # Save per-utterance results
    utt_results_file = os.path.join(args.outdir, "mer_per_utterance.txt")
    with open(utt_results_file, "w") as f:
        f.write("Utterance-level MER Results\n")
        f.write("=" * 50 + "\n")
        for utt_id, results in utt_results.items():
            overall_wer = results["overall"].get("wer", 0.0)
            f.write(f"{utt_id} {overall_wer:.4f}\n")

    logging.info(f"Per-utterance results saved to {utt_results_file}")


if __name__ == "__main__":
    main()
