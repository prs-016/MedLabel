---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:749
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
    ['What are the symptoms and treatment for acute spironolactone overdose?', '10 OVERDOSAGE The oral LD 50 of spironolactone is greater than 1000 mg/kg in mice, rats, and rabbits. Acute overdosage of spironolactone may be manifested by drowsiness, mental confusion, maculopapular or erythematous rash, nausea, vomiting, dizziness, or diarrhea. Rarely, instances of hyponatremia, hyperkalemia, or hepatic coma may occur in patients with severe liver disease, but these are unlikely due to acute overdosage. Hyperkalemia may occur, especially in patients with impaired renal function. Treatment: Induce vomiting or evacuate the stomach by lavage. There is no specific antidote. Treatment is supportive to maintain hydration, electrolyte balance, and vital functions. Patients who have renal impairment may develop hyperkalemia. In such cases, discontinue spironolactone.'],
    ['Does duloxetine cause palpitations or vision changes that would contraindicate its use in patients with cardiac disorders?', 'Adverse Reactions in Pooled MDD and GAD Trials in Adults Table 3 displays the incidence of adverse reactions in MDD and GAD placebo-controlled adult trials that occurred in 2% or more of duloxetine-treated patients and with an incidence greater than placebo-treated patients. Table 3: Adverse Reactions: Incidence of 2% or More and Greater than Placebo in MDD and GAD Placebo-Controlled Trials in Adults The inclusion of an event in the table is determined based on the percentages before rounding; however, the percentages displayed in the table are rounded to the nearest integer. , For GAD, there were no adverse reactions that were significantly different between treatments in adults ≥65 years that were also not significant in the adults <65 years. System Organ Class / Adverse Reaction Percentage of Patients Reporting Reaction Duloxetine delayed-release capsules (N=4797) Placebo (N=3303) Cardiac Disorders Palpitations 2 1 Eye Disorders Vision blurred 3 1 Gastrointestinal Disorders Nausea Events for which there was a significant dose-dependent relationship in fixed-dose studies, excluding three MDD studies which did not have a placebo lead-in period or dose titration. 23 8 Dry mouth 14 6 Constipation 9 4 Diarrhea 9 6 Abdominal pain Includes abdominal pain upper, abdominal pain lower, abdominal'],
    ['What are the signs and symptoms of overdosage with salmeterol or fluticasone propionate?', '10 OVERDOSAGE This product contains both fluticasone propionate and salmeterol; therefore, the risks associated with overdosage for the individual components described below apply to Fluticasone Propionate/Salmeterol MDPI. Treatment of overdosage consists of discontinuation of Fluticasone Propionate/Salmeterol MDPI together with institution of appropriate symptomatic and/or supportive therapy. The judicious use of a cardioselective beta‑receptor blocker may be considered, bearing in mind that such medication can produce bronchospasm. Cardiac monitoring is recommended in cases of overdosage. Fluticasone propionate Chronic overdosage of fluticasone propionate may result in signs/symptoms of hypercorticism [see Warnings and Precautions ( 5.7 )] . Salmeterol The expected signs and symptoms with overdosage of salmeterol are those of excessive beta‑adrenergic stimulation and/or occurrence or exaggeration of any of the signs and symptoms of beta‑adrenergic stimulation (e.g., seizures, angina, hypertension or hypotension, tachycardia with rates up to 200 beats/min, arrhythmias, nervousness, headache, tremor, muscle cramps, dry mouth, palpitation, nausea, dizziness, fatigue, malaise, insomnia, hyperglycemia, hypokalemia, metabolic acidosis). Overdosage with salmeterol can lead to clinically significant prolongation of the QTc interval, which can produce ventricular arrhythmias. As with all inhaled sympathomimetic medicines, cardiac arrest and even death may be associated with an overd'],
    ['What are the contraindications for gabapentin with other CNS depressants?', '10 OVERDOSAGE Signs of acute toxicity in animals included ataxia, labored breathing, ptosis, sedation, hypoactivity, or excitation. Acute oral overdoses of gabapentin have been reported. Symptoms have included double vision, tremor, slurred speech, drowsiness, altered mental status, dizziness, lethargy, and diarrhea. Fatal respiratory depression has been reported with gabapentin overdose, alone and in combination with other CNS depressants. Gabapentin can be removed by hemodialysis. If overexposure occurs, call your poison control center at 1-800-222-1222.'],
    ['What are the contraindications for tramadol hydrochloride extended-release tablets?', '4 CONTRAINDICATIONS Tramadol hydrochloride extended-release tablets are contraindicated for: all children younger than 12 years of age [see Warnings and Precautions (5.4)] post-operative management in children younger than 18 years of age following tonsillectomy and/or adenoidectomy [see Warnings and Precautions (5.4)] . Tramadol hydrochloride extended-release tablets are also contraindicated in patients with: Significant respiratory depression [see Warnings and Precautions (5.3)] Acute or severe bronchial asthma in an unmonitored setting or in the absence of resuscitative equipment [see Warnings and Precautions (5.12)] Known or suspected gastrointestinal obstruction, including paralytic ileus [see Warnings and Precautions (5.15)] Hypersensitivity to tramadol (e.g., anaphylaxis) [see Warnings and Precautions (5.17), Adverse Reactions (6.2)] Concurrent use of monoamine oxidase inhibitors (MAOIs) or use within the last 14 days [see Drug Interactions (7)] . Children younger than 12 years of age (4) Postoperative management in children younger than 18 years of age following tonsillectomy and/or adenoidectomy. (4) Significant respiratory depression (4) Acute or severe bronchial asthma in an unmonitored setting or in absence of resuscitative equipment (4) Known or suspected gastrointestinal obstruction, including paralytic ileus (4) Hypersensitivity to tramadol (4) Concurrent use of monoamine oxidase inhibitors (MAOIs) or use within the last 14 days (4)'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What are the symptoms and treatment for acute spironolactone overdose?',
    [
        '10 OVERDOSAGE The oral LD 50 of spironolactone is greater than 1000 mg/kg in mice, rats, and rabbits. Acute overdosage of spironolactone may be manifested by drowsiness, mental confusion, maculopapular or erythematous rash, nausea, vomiting, dizziness, or diarrhea. Rarely, instances of hyponatremia, hyperkalemia, or hepatic coma may occur in patients with severe liver disease, but these are unlikely due to acute overdosage. Hyperkalemia may occur, especially in patients with impaired renal function. Treatment: Induce vomiting or evacuate the stomach by lavage. There is no specific antidote. Treatment is supportive to maintain hydration, electrolyte balance, and vital functions. Patients who have renal impairment may develop hyperkalemia. In such cases, discontinue spironolactone.',
        'Adverse Reactions in Pooled MDD and GAD Trials in Adults Table 3 displays the incidence of adverse reactions in MDD and GAD placebo-controlled adult trials that occurred in 2% or more of duloxetine-treated patients and with an incidence greater than placebo-treated patients. Table 3: Adverse Reactions: Incidence of 2% or More and Greater than Placebo in MDD and GAD Placebo-Controlled Trials in Adults The inclusion of an event in the table is determined based on the percentages before rounding; however, the percentages displayed in the table are rounded to the nearest integer. , For GAD, there were no adverse reactions that were significantly different between treatments in adults ≥65 years that were also not significant in the adults <65 years. System Organ Class / Adverse Reaction Percentage of Patients Reporting Reaction Duloxetine delayed-release capsules (N=4797) Placebo (N=3303) Cardiac Disorders Palpitations 2 1 Eye Disorders Vision blurred 3 1 Gastrointestinal Disorders Nausea Events for which there was a significant dose-dependent relationship in fixed-dose studies, excluding three MDD studies which did not have a placebo lead-in period or dose titration. 23 8 Dry mouth 14 6 Constipation 9 4 Diarrhea 9 6 Abdominal pain Includes abdominal pain upper, abdominal pain lower, abdominal',
        '10 OVERDOSAGE This product contains both fluticasone propionate and salmeterol; therefore, the risks associated with overdosage for the individual components described below apply to Fluticasone Propionate/Salmeterol MDPI. Treatment of overdosage consists of discontinuation of Fluticasone Propionate/Salmeterol MDPI together with institution of appropriate symptomatic and/or supportive therapy. The judicious use of a cardioselective beta‑receptor blocker may be considered, bearing in mind that such medication can produce bronchospasm. Cardiac monitoring is recommended in cases of overdosage. Fluticasone propionate Chronic overdosage of fluticasone propionate may result in signs/symptoms of hypercorticism [see Warnings and Precautions ( 5.7 )] . Salmeterol The expected signs and symptoms with overdosage of salmeterol are those of excessive beta‑adrenergic stimulation and/or occurrence or exaggeration of any of the signs and symptoms of beta‑adrenergic stimulation (e.g., seizures, angina, hypertension or hypotension, tachycardia with rates up to 200 beats/min, arrhythmias, nervousness, headache, tremor, muscle cramps, dry mouth, palpitation, nausea, dizziness, fatigue, malaise, insomnia, hyperglycemia, hypokalemia, metabolic acidosis). Overdosage with salmeterol can lead to clinically significant prolongation of the QTc interval, which can produce ventricular arrhythmias. As with all inhaled sympathomimetic medicines, cardiac arrest and even death may be associated with an overd',
        '10 OVERDOSAGE Signs of acute toxicity in animals included ataxia, labored breathing, ptosis, sedation, hypoactivity, or excitation. Acute oral overdoses of gabapentin have been reported. Symptoms have included double vision, tremor, slurred speech, drowsiness, altered mental status, dizziness, lethargy, and diarrhea. Fatal respiratory depression has been reported with gabapentin overdose, alone and in combination with other CNS depressants. Gabapentin can be removed by hemodialysis. If overexposure occurs, call your poison control center at 1-800-222-1222.',
        '4 CONTRAINDICATIONS Tramadol hydrochloride extended-release tablets are contraindicated for: all children younger than 12 years of age [see Warnings and Precautions (5.4)] post-operative management in children younger than 18 years of age following tonsillectomy and/or adenoidectomy [see Warnings and Precautions (5.4)] . Tramadol hydrochloride extended-release tablets are also contraindicated in patients with: Significant respiratory depression [see Warnings and Precautions (5.3)] Acute or severe bronchial asthma in an unmonitored setting or in the absence of resuscitative equipment [see Warnings and Precautions (5.12)] Known or suspected gastrointestinal obstruction, including paralytic ileus [see Warnings and Precautions (5.15)] Hypersensitivity to tramadol (e.g., anaphylaxis) [see Warnings and Precautions (5.17), Adverse Reactions (6.2)] Concurrent use of monoamine oxidase inhibitors (MAOIs) or use within the last 14 days [see Drug Interactions (7)] . Children younger than 12 years of age (4) Postoperative management in children younger than 18 years of age following tonsillectomy and/or adenoidectomy. (4) Significant respiratory depression (4) Acute or severe bronchial asthma in an unmonitored setting or in absence of resuscitative equipment (4) Known or suspected gastrointestinal obstruction, including paralytic ileus (4) Hypersensitivity to tramadol (4) Concurrent use of monoamine oxidase inhibitors (MAOIs) or use within the last 14 days (4)',
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

* Size: 749 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 749 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                          | label                                                          |
  |:--------|:------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                              | float                                                          |
  | details | <ul><li>min: 39 characters</li><li>mean: 97.93 characters</li><li>max: 213 characters</li></ul> | <ul><li>min: 130 characters</li><li>mean: 1030.54 characters</li><li>max: 1500 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.33</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What are the symptoms and treatment for acute spironolactone overdose?</code>                                                     | <code>10 OVERDOSAGE The oral LD 50 of spironolactone is greater than 1000 mg/kg in mice, rats, and rabbits. Acute overdosage of spironolactone may be manifested by drowsiness, mental confusion, maculopapular or erythematous rash, nausea, vomiting, dizziness, or diarrhea. Rarely, instances of hyponatremia, hyperkalemia, or hepatic coma may occur in patients with severe liver disease, but these are unlikely due to acute overdosage. Hyperkalemia may occur, especially in patients with impaired renal function. Treatment: Induce vomiting or evacuate the stomach by lavage. There is no specific antidote. Treatment is supportive to maintain hydration, electrolyte balance, and vital functions. Patients who have renal impairment may develop hyperkalemia. In such cases, discontinue spironolactone.</code>                                                                                                                                                                                                                      | <code>1.0</code> |
  | <code>Does duloxetine cause palpitations or vision changes that would contraindicate its use in patients with cardiac disorders?</code> | <code>Adverse Reactions in Pooled MDD and GAD Trials in Adults Table 3 displays the incidence of adverse reactions in MDD and GAD placebo-controlled adult trials that occurred in 2% or more of duloxetine-treated patients and with an incidence greater than placebo-treated patients. Table 3: Adverse Reactions: Incidence of 2% or More and Greater than Placebo in MDD and GAD Placebo-Controlled Trials in Adults The inclusion of an event in the table is determined based on the percentages before rounding; however, the percentages displayed in the table are rounded to the nearest integer. , For GAD, there were no adverse reactions that were significantly different between treatments in adults ≥65 years that were also not significant in the adults <65 years. System Organ Class / Adverse Reaction Percentage of Patients Reporting Reaction Duloxetine delayed-release capsules (N=4797) Placebo (N=3303) Cardiac Disorders Palpitations 2 1 Eye Disorders Vision blurred 3 1 Gastrointestinal Disorders Nausea E...</code> | <code>0.0</code> |
  | <code>What are the signs and symptoms of overdosage with salmeterol or fluticasone propionate?</code>                                   | <code>10 OVERDOSAGE This product contains both fluticasone propionate and salmeterol; therefore, the risks associated with overdosage for the individual components described below apply to Fluticasone Propionate/Salmeterol MDPI. Treatment of overdosage consists of discontinuation of Fluticasone Propionate/Salmeterol MDPI together with institution of appropriate symptomatic and/or supportive therapy. The judicious use of a cardioselective beta‑receptor blocker may be considered, bearing in mind that such medication can produce bronchospasm. Cardiac monitoring is recommended in cases of overdosage. Fluticasone propionate Chronic overdosage of fluticasone propionate may result in signs/symptoms of hypercorticism [see Warnings and Precautions ( 5.7 )] . Salmeterol The expected signs and symptoms with overdosage of salmeterol are those of excessive beta‑adrenergic stimulation and/or occurrence or exaggeration of any of the signs and symptoms of beta‑adrenergic stimulation (e.g., seizures, angina,...</code> | <code>1.0</code> |
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
- `fp16`: False
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