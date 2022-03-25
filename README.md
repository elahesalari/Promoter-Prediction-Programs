# Promoter-Prediction-Programs

The promoter region is located near the transcription start sites which regulates the 
transcription initiation of the gene by controlling the binding of RNA polymerase.
Thus, promoter region recognition is an important area of interest in the field of 
bioinformatics. During the past years, many new promoter prediction programs
(PPPs) have been emerged. PPPs aim to identify promoter regions in a genome
using computational methods. Promoter prediction is a supervised learning
problem which contains three main steps to extract features:
1. CpG islands 
2. Structural features 
3. Content features

<b> Task:</b>
<br/>
You are asked to extract suitable features in order to train a discriminative
model on dataset to classify promoter and non-promoter sequences

In the implementations, you should notice: <br/>
➢ Related paper "Large-scale structural analysis of the core promoter in mammalian and plant genomes" is attached in order to get idea about feature definitions. <br/>
➢ You can use the conversion table to calculate structural features. <br/>
➢ You can use any feature selection method to reduce dimensionality. <br/>
➢ Running time is important so, you should choose suitable classifier.<br/>
➢ Implement 5-fold CV and calculate precision, recall and F-score to evaluate
your method. <br/>

<b>Datasets:</b>
<br/>
The dataset has 60,000 2-class samples which contain 30,000 promoter samples 
and 30,000 exons, UTR3s and introns as non-promoter samples.
