# Task 4 Diversity and Style Steering Summary

- Images evaluated for diversity: 200
- Captions per image: 5 (nucleus sampling, p=0.9)
- Mean caption diversity (unique ngrams / total ngrams): 0.7373
- Mean pre-beam hidden-state diversity: 0.8460

## Diverse vs Repetitive Image Types (keyword proxy)

- Low-diversity keywords: [('man', 38), ('sitting', 21), ('standing', 20), ('group', 15), ('people', 11), ('table', 10), ('holding', 10), ('dog', 10), ('cake', 9), ('next', 8), ('field', 8), ('person', 8)]
- High-diversity keywords: [('two', 20), ('sitting', 13), ('man', 10), ('table', 9), ('front', 8), ('standing', 8), ('young', 8), ('dog', 7), ('people', 7), ('next', 6), ('three', 6), ('holding', 5)]

## Steering Effect (Mean Caption Length)

- Baseline: 10.28
- Steered Short: 10.79
- Steered Long: 10.45
- Steered Detailed: 10.11

## Interpretation

- If steered-long > baseline > steered-short, steering direction is working.
- Detailed steering should increase descriptive tokens and relational words.
