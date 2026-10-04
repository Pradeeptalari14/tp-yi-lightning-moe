"""Bilingual token-efficiency estimator.

Estimates token counts for English vs. Chinese text under a generic tokenizer
and a CJK-optimised tokenizer, to size KV-cache and cost budgets. Ratios are
heuristic planning numbers, not measurements of a specific production tokenizer.
"""

from typing import Dict


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def estimate_tokens(text: str, cjk_optimised: bool) -> int:
    cjk_chars = sum(1 for c in text if is_cjk(c))
    latin_words = len("".join(" " if is_cjk(c) else c for c in text).split())
    # Generic BPE: ~1.5 tokens per CJK char; CJK-optimised vocab: ~0.7.
    cjk_rate = 0.7 if cjk_optimised else 1.5
    return max(1, round(cjk_chars * cjk_rate + latin_words * 1.3))


def compare(text: str) -> Dict[str, float]:
    generic = estimate_tokens(text, cjk_optimised=False)
    optimised = estimate_tokens(text, cjk_optimised=True)
    return {"generic_tokens": generic, "cjk_optimised_tokens": optimised,
            "compression": round(generic / optimised, 2)}


def main() -> None:
    zh = "混合专家模型通过稀疏激活降低推理成本，同时保持高质量的双语推理能力。"
    en = "Mixture of experts models reduce inference cost through sparse activation."
    zh_stats, en_stats = compare(zh), compare(en)
    print("Chinese:", zh_stats)
    print("English:", en_stats)
    assert zh_stats["compression"] > 1.5
    assert en_stats["compression"] == 1.0
    print("Tokenizer estimator validation passed.")


if __name__ == "__main__":
    main()
