# Devanagari Tokenization for Hindi and Sanskrit

An NLP project that studies the effect of **linguistically informed pretokenization** and **script-aware subword tokenization** on Devanagari script languages, with a focus on Hindi and Sanskrit.

---

## Abstract

Tokenization plays a critical role in Natural Language Processing (NLP) pipelines. However, most widely used tokenization algorithms are designed for English and do not account for the linguistic properties of **Indic languages written in the Devanagari script**.  
This project investigates tokenization strategies that better preserve **akshara structure**, **morphological boundaries**, and **sandhi phenomena**, and evaluates their impact on downstream NLP tasks.

---

## Motivation

Devanagari script languages exhibit:
- Akshara-based orthography
- Rich morphology and compounding
- Sandhi (word fusion without explicit boundaries)

Standard subword tokenizers such as Byte Pair Encoding (BPE) often over-segment such text, leading to loss of semantic and syntactic information.  
This motivates the use of **pretok­enization** and **script-aware tokenization techniques**.

---

## NLP Techniques Used

This project incorporates the following core NLP components:

### 🔹 Pretokenization
- Linguistic preprocessing step applied before subword learning
- Aims to split text into meaningful morpheme-like units
- Reduces noise and over-segmentation in downstream tokenization

### 🔹 Subword Tokenization
- **Byte Pair Encoding (BPE)**  
- **Unigram Language Model**
- A modified, **Unicode-aware BPE variant** to better preserve Devanagari script structure

### 🔹 Script-Aware Processing
- Exploits Unicode character categories
- Encourages merges that form valid aksharas
- Discourages linguistically invalid splits

### 🔹 Sequence-to-Sequence Modeling
- Transformer-based neural machine translation
- Used as a downstream task to evaluate tokenization quality

### 🔹 Evaluation
- Quantitative evaluation using **BLEU score**
- Comparative analysis across tokenization strategies

---

## Methodology

The experimental pipeline consists of the following stages:

1. **Text Pretokenization**  
   Linguistic segmentation of raw Hindi and Sanskrit text.

2. **Subword Tokenization**  
   Learning token vocabularies using different tokenization algorithms.

3. **Model Training**  
   Training sequence-to-sequence models using the generated tokenizations.

4. **Evaluation & Analysis**  
   Measuring translation quality and analyzing token stability.

---

## Tools and Frameworks

- **Python**
- **Natural Language Processing (NLP)**
- **Subword Tokenization Algorithms**
- **Transformer Models**
- **BLEU Score Evaluation**

---

