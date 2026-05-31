---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:358
- loss:BinaryCrossEntropyLoss
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model trained using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['What is the risk of SJS/TEN with carbamazepine and its association with the HLA-B*1502 allele in Asian populations?', 'WARNINGS Serious Dermatologic Reactions Serious and sometimes fatal dermatologic reactions, including toxic epidermal necrolysis (TEN) and Stevens-Johnson syndrome (SJS), have been reported with carbamazepine treatment. The risk of these events is estimated to be about 1 to 6 per 10,000 new users in countries with mainly Caucasian populations. However, the risk in some Asian countries is estimated to be about 10 times higher. Carbamazepine should be discontinued at the first sign of a rash, unless the rash is clearly not drug-related. If signs or symptoms suggest SJS/TEN, use of this drug should not be resumed and alternative therapy should be considered. SJS/TEN and HLA-B*1502 Allele Retrospective case-control studies have found that in patients of Chinese ancestry there is a strong association between the risk of developing SJS/TEN with carbamazepine treatment and the presence of an inherited variant of the HLA-B gene, HLA-B*1502. The occurrence of higher rates of these reactions in countries with higher frequencies of this allele suggests that the risk may be increased in allele-positive individuals of any ethnicity. Across Asian populations, notable variation exists in the prevalence of HLA-B*1502. Greater than 15% of the population is reported positive in Hong Kong, Thailand, Malaysia, and parts of the Philippines, compared to about 10% in Taiwan and 4% in North China. South Asians, including Indians, appear to have intermediate prevalence of HLA-B*1502, averaging 2% to '],
    ['What are the warnings for heart failure or pancreatitis when using ZITUVIMET?', '4 CONTRAINDICATIONS XARELTO is contraindicated in patients with: active pathological bleeding [see Warnings and Precautions (5.2) ] severe hypersensitivity reaction to XARELTO (e.g., anaphylactic reactions) [see Adverse Reactions (6.2) ] Active pathological bleeding ( 4 ) Severe hypersensitivity reaction to XARELTO ( 4 )'],
    ['What adverse reactions including anaphylaxis have been reported with XARELTO?', '4 CONTRAINDICATIONS XARELTO is contraindicated in patients with: active pathological bleeding [see Warnings and Precautions (5.2) ] severe hypersensitivity reaction to XARELTO (e.g., anaphylactic reactions) [see Adverse Reactions (6.2) ] Active pathological bleeding ( 4 ) Severe hypersensitivity reaction to XARELTO ( 4 )'],
    ['What symptoms or signs occur with gabapentin overdose or acute toxicity?', '10 OVERDOSAGE Signs of acute toxicity in animals included ataxia, labored breathing, ptosis, sedation, hypoactivity, or excitation. Acute oral overdoses of gabapentin have been reported. Symptoms have included double vision, tremor, slurred speech, drowsiness, altered mental status, dizziness, lethargy, and diarrhea. Fatal respiratory depression has been reported with gabapentin overdose, alone and in combination with other CNS depressants. Gabapentin can be removed by hemodialysis. If overexposure occurs, call your poison control center at 1-800-222-1222.'],
    ['What risks are associated with stopping apixaban early or using it with spinal procedures?', '5 WARNINGS AND PRECAUTIONS Hyperkalemia: Monitor serum potassium within one week of initiation and regularly thereafter ( 5.1 ). Hypotension and Worsening Renal Function: Monitor volume status and renal function periodically ( 5.2 ). Electrolyte and Metabolic Abnormalities: Monitor serum electrolytes, uric acid and blood glucose periodically ( 5.3 ). Gynecomastia: Spironolactone can cause gynecomastia ( 5.4 ). 5.1 Hyperkalemia Spironolactone can cause hyperkalemia. This risk is increased by impaired renal function or concomitant potassium supplementation, potassium-containing salt substitutes or drugs that increase potassium, such as angiotensin converting enzyme inhibitors and angiotensin receptor blockers [see Drug Interactions (7.1) ] . Monitor serum potassium within 1 week of initiation or titration of spironolactone and regularly thereafter. More frequent monitoring may be needed when spironolactone is given with other drugs that cause hyperkalemia or in patients with impaired renal function. If hyperkalemia occurs, decrease the dose or discontinue spironolactone and treat hyperkalemia. 5.2 Hypotension and Worsening Renal Function Excessive diuresis may cause symptomatic dehydration, hypotension and worsening renal function, particularly in salt-depleted patients or those taking angiotensin converting enzyme inhibitors and angiotensin II receptor blockers. Worsening of renal function can also occur with concomitant use of nephrotoxic drugs (e.g., aminoglycosides, cisplat'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What is the risk of SJS/TEN with carbamazepine and its association with the HLA-B*1502 allele in Asian populations?',
    [
        'WARNINGS Serious Dermatologic Reactions Serious and sometimes fatal dermatologic reactions, including toxic epidermal necrolysis (TEN) and Stevens-Johnson syndrome (SJS), have been reported with carbamazepine treatment. The risk of these events is estimated to be about 1 to 6 per 10,000 new users in countries with mainly Caucasian populations. However, the risk in some Asian countries is estimated to be about 10 times higher. Carbamazepine should be discontinued at the first sign of a rash, unless the rash is clearly not drug-related. If signs or symptoms suggest SJS/TEN, use of this drug should not be resumed and alternative therapy should be considered. SJS/TEN and HLA-B*1502 Allele Retrospective case-control studies have found that in patients of Chinese ancestry there is a strong association between the risk of developing SJS/TEN with carbamazepine treatment and the presence of an inherited variant of the HLA-B gene, HLA-B*1502. The occurrence of higher rates of these reactions in countries with higher frequencies of this allele suggests that the risk may be increased in allele-positive individuals of any ethnicity. Across Asian populations, notable variation exists in the prevalence of HLA-B*1502. Greater than 15% of the population is reported positive in Hong Kong, Thailand, Malaysia, and parts of the Philippines, compared to about 10% in Taiwan and 4% in North China. South Asians, including Indians, appear to have intermediate prevalence of HLA-B*1502, averaging 2% to ',
        '4 CONTRAINDICATIONS XARELTO is contraindicated in patients with: active pathological bleeding [see Warnings and Precautions (5.2) ] severe hypersensitivity reaction to XARELTO (e.g., anaphylactic reactions) [see Adverse Reactions (6.2) ] Active pathological bleeding ( 4 ) Severe hypersensitivity reaction to XARELTO ( 4 )',
        '4 CONTRAINDICATIONS XARELTO is contraindicated in patients with: active pathological bleeding [see Warnings and Precautions (5.2) ] severe hypersensitivity reaction to XARELTO (e.g., anaphylactic reactions) [see Adverse Reactions (6.2) ] Active pathological bleeding ( 4 ) Severe hypersensitivity reaction to XARELTO ( 4 )',
        '10 OVERDOSAGE Signs of acute toxicity in animals included ataxia, labored breathing, ptosis, sedation, hypoactivity, or excitation. Acute oral overdoses of gabapentin have been reported. Symptoms have included double vision, tremor, slurred speech, drowsiness, altered mental status, dizziness, lethargy, and diarrhea. Fatal respiratory depression has been reported with gabapentin overdose, alone and in combination with other CNS depressants. Gabapentin can be removed by hemodialysis. If overexposure occurs, call your poison control center at 1-800-222-1222.',
        '5 WARNINGS AND PRECAUTIONS Hyperkalemia: Monitor serum potassium within one week of initiation and regularly thereafter ( 5.1 ). Hypotension and Worsening Renal Function: Monitor volume status and renal function periodically ( 5.2 ). Electrolyte and Metabolic Abnormalities: Monitor serum electrolytes, uric acid and blood glucose periodically ( 5.3 ). Gynecomastia: Spironolactone can cause gynecomastia ( 5.4 ). 5.1 Hyperkalemia Spironolactone can cause hyperkalemia. This risk is increased by impaired renal function or concomitant potassium supplementation, potassium-containing salt substitutes or drugs that increase potassium, such as angiotensin converting enzyme inhibitors and angiotensin receptor blockers [see Drug Interactions (7.1) ] . Monitor serum potassium within 1 week of initiation or titration of spironolactone and regularly thereafter. More frequent monitoring may be needed when spironolactone is given with other drugs that cause hyperkalemia or in patients with impaired renal function. If hyperkalemia occurs, decrease the dose or discontinue spironolactone and treat hyperkalemia. 5.2 Hypotension and Worsening Renal Function Excessive diuresis may cause symptomatic dehydration, hypotension and worsening renal function, particularly in salt-depleted patients or those taking angiotensin converting enzyme inhibitors and angiotensin II receptor blockers. Worsening of renal function can also occur with concomitant use of nephrotoxic drugs (e.g., aminoglycosides, cisplat',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 358 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 358 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                          | label                                                          |
  |:--------|:------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                              | float                                                          |
  | details | <ul><li>min: 43 characters</li><li>mean: 91.56 characters</li><li>max: 148 characters</li></ul> | <ul><li>min: 130 characters</li><li>mean: 1140.79 characters</li><li>max: 1500 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.34</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                       | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What is the risk of SJS/TEN with carbamazepine and its association with the HLA-B*1502 allele in Asian populations?</code> | <code>WARNINGS Serious Dermatologic Reactions Serious and sometimes fatal dermatologic reactions, including toxic epidermal necrolysis (TEN) and Stevens-Johnson syndrome (SJS), have been reported with carbamazepine treatment. The risk of these events is estimated to be about 1 to 6 per 10,000 new users in countries with mainly Caucasian populations. However, the risk in some Asian countries is estimated to be about 10 times higher. Carbamazepine should be discontinued at the first sign of a rash, unless the rash is clearly not drug-related. If signs or symptoms suggest SJS/TEN, use of this drug should not be resumed and alternative therapy should be considered. SJS/TEN and HLA-B*1502 Allele Retrospective case-control studies have found that in patients of Chinese ancestry there is a strong association between the risk of developing SJS/TEN with carbamazepine treatment and the presence of an inherited variant of the HLA-B gene, HLA-B*1502. The occurrence of higher rates of these reactions in ...</code> | <code>1.0</code> |
  | <code>What are the warnings for heart failure or pancreatitis when using ZITUVIMET?</code>                                       | <code>4 CONTRAINDICATIONS XARELTO is contraindicated in patients with: active pathological bleeding [see Warnings and Precautions (5.2) ] severe hypersensitivity reaction to XARELTO (e.g., anaphylactic reactions) [see Adverse Reactions (6.2) ] Active pathological bleeding ( 4 ) Severe hypersensitivity reaction to XARELTO ( 4 )</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>0.0</code> |
  | <code>What adverse reactions including anaphylaxis have been reported with XARELTO?</code>                                       | <code>4 CONTRAINDICATIONS XARELTO is contraindicated in patients with: active pathological bleeding [see Warnings and Precautions (5.2) ] severe hypersensitivity reaction to XARELTO (e.g., anaphylactic reactions) [see Adverse Reactions (6.2) ] Active pathological bleeding ( 4 ) Severe hypersensitivity reaction to XARELTO ( 4 )</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.9.6
- Sentence Transformers: 5.1.2
- Transformers: 4.57.6
- PyTorch: 2.8.0
- Accelerate: 1.10.1
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->