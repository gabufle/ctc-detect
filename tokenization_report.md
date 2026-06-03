# Tokenization Report

## Data Shape
- Original cells: 11156
- Original genes: 7637
- Genes mapped to Ensembl IDs: 7517
- Genes unmapped (dropped): 120
- Final shape: (11156, 7517)

## Splits
- train: 7808 cells -> 7808 tokenized examples
- val: 1674 cells -> 1674 tokenized examples
- test: 1674 cells -> 1674 tokenized examples

## Tokenized Data Verification
### train
- Number of examples: 7808
- Average sequence length: 1101.4
- First example keys: ['input_ids', 'length']
- First example input_ids length: 1141
- First 10 token IDs: [2, 7245, 18810, 12014, 15797, 3865, 14528, 1642, 12170, 13339]
### val
- Number of examples: 1674
- Average sequence length: 1122.3
- First example keys: ['input_ids', 'length']
- First example input_ids length: 795
- First 10 token IDs: [2, 1753, 115, 15262, 13789, 981, 17327, 10064, 3155, 15947]
### test
- Number of examples: 1674
- Average sequence length: 1102.6
- First example keys: ['input_ids', 'length']
- First example input_ids length: 986
- First 10 token IDs: [2, 115, 17327, 17816, 10064, 327, 4922, 19086, 18815, 15947]

## Vocabulary Coverage
- Token dictionary: token_dictionary_gc104M.pkl
- Vocabulary size: 20275
- Genes in our data (Ensembl IDs): 7517
- Genes in vocabulary: 7442 (99.00%)
- Genes dropped (not in vocabulary): 75

### Genes without Ensembl mapping (120)
- C1orf109
- C1orf112
- C1orf122
- C1orf123
- C1orf131
- C1orf162
- C1orf174
- C1orf198
- C1orf21
- C1orf35
- C1orf43
- C1orf50
- C1orf52
- C1orf56
- C2orf49
- C2orf68
- C2orf69
- C2orf76
- C2orf88
- C3orf38
- C3orf52
- C3orf58
- C3orf62
- C4orf3
- C4orf48
- NOTCH2NL
- NPPA-AS1
- QARS
- SARS
- SEPT2
- ... and 90 more

## Notes
- Tokenization performed using Geneformer's TranscriptomeTokenizer (model_version='V2').
- Data was normalized (log1p) from preprocessing step.
- Gene symbol -> Ensembl ID mapping via ensembl_mapping_dict_gc104M.pkl.
- Genes without Ensembl IDs were filtered out before tokenization.
- n_counts column computed from data matrix and added to adata.obs.
- tokenize_anndata() used per split, then create_dataset() to build HF Dataset.
- Each split saved as HuggingFace Dataset via save_to_disk().