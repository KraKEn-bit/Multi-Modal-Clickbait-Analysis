# VTCF Evaluation Report

Automated evaluation summary for the Visual-Temporal Contradiction Framework.

## Ablation Comparison

| Condition | F1 Det | F1 Attr | TDS Corr | p-value vs Full |
|-----------|--------|---------|----------|-----------------|
| Text Only | 0.9038 | 0.0000 | nan | 0.0000 |
| Vision Only | 0.9938 | 0.0000 | 0.5944 | 1.0000 |
| Full VTCF | 0.9950 | 0.0000 | 0.5361 | --- |

## Ambiguous Text Subset

Samples where the text-only model confidence was between 0.45 and 0.55.

- Subset size: **10**
- Text-Only F1: **0.3333**
- VTCF F1: **1.0000**
- Delta (VTCF - Text-Only): **+0.6667**

## McNemar's Test

McNemar's test assesses whether paired prediction differences between
conditions are statistically significant on the same test set (p < 0.05).

- **text_only vs Full VTCF**: chi2=65.6203, p=0.0000, significant=True
- **vision_only vs Full VTCF**: chi2=0.0000, p=1.0000, significant=False

## Notes

- Higher TDS correlation with clickbait labels supports the bait-and-switch hypothesis.
- A positive ambiguous-subset delta indicates the visual branch adds signal when text is unreliable.
