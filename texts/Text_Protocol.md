# Text Generation and Manual Checking

## Generation prompt

We used LLM to generate a description for each land-cover category. In the following prompt template, `[CLASS]` was replaced with the corresponding class name.

> Given the land-cover class [CLASS], generate a concise description integrating its general semantic definition, spectral characteristics, and spatial distribution patterns in hyperspectral images.

## Manual checking

Before text encoding, we manually reviewed each generated description using the following two criteria.

1. **Consistency with the class label.** We checked that the description referred to the land-cover category indicated by the dataset label and preserved the distinction conveyed by that label. For example, the descriptions of *Soybean notill* and *Soybean mintill* should distinguish between no tillage and minimum tillage.
2. **Semantic consistency.** We checked that the general definition and the stated spectral and spatial characteristics were plausible for the corresponding class. For example, the Chikusei description of *paved ground* refers to concrete or brick surfaces and describes smooth spectral variation and regular spatial patterns.

## Text files

The files `CH_text.txt`, `IP_text.txt`, `SA_text.txt`, and `UP_text.txt` each contain the category-level descriptions in class order and the corresponding dataset-level description, formed by concatenating the category-level descriptions in that order.
