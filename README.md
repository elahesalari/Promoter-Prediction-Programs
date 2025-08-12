# Promoter Prediction Programs

Promoter regions, located near transcription start sites, regulate gene transcription by controlling the binding of RNA polymerase. Accurate identification of promoter regions is a critical task in bioinformatics. Over recent years, many computational promoter prediction programs (PPPs) have been developed to address this supervised learning problem.

---

## Project Overview

The promoter prediction task involves extracting meaningful features from genomic sequences and training a discriminative model to classify promoter versus non-promoter sequences. The three primary categories of features extracted are:

1. **CpG Islands**  
2. **Structural Features**  
3. **Content Features**

---

## Implementation Details

- Feature extraction was guided by the related paper *“Large-scale structural analysis of the core promoter in mammalian and plant genomes”*, which provides definitions and insights for structural features.  
- Structural features were computed using a conversion table approach.  
- Dimensionality reduction was performed via feature selection methods to improve model efficiency.  
- Considering runtime constraints, an appropriate classifier was selected to balance accuracy and computational cost.  
- The model was evaluated using **5-fold cross-validation**, calculating **precision**, **recall**, and **F-score** to assess performance.

---

## Dataset

- The dataset consists of 60,000 samples divided evenly into two classes:  
  - 30,000 promoter sequences  
  - 30,000 non-promoter sequences (including exons, 3’ UTRs, and introns)

