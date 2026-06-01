---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:897
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
    ['What is the recommended antidote dose for clopidogrel overdose in adult humans?', '10 OVERDOSAGE Platelet inhibition by clopidogrel is irreversible and will last for the life of the platelet. Overdose following clopidogrel administration may result in bleeding complications. A single oral dose of clopidogrel at 1500 or 2000 mg/kg was lethal to mice and to rats and at 3000 mg/kg to baboons. Symptoms of acute toxicity were vomiting, prostration, difficult breathing, and gastrointestinal hemorrhage in animals. Based on biological plausibility, platelet transfusion may restore clotting ability.'],
    ['What are the adverse effects of rosuvastatin when combined with cyclosporine in renal impairment?', 'Precautions ( 5.1 ), Use in Specific Populations ( 8.8 ), and Clinical Pharmacology ( 12.3 )]. 2.5 Recommended Dosage in Patients with Renal Impairment In patients with severe renal impairment (CL cr less than 30 mL/min/1.73 m 2 ) not on hemodialysis, the recommended starting dosage is 5 mg once daily and should not exceed 10 mg once daily [see Warnings and Precautions ( 5.1 ) and Use in Specific Populations ( 8.6 )]. There are no dosage adjustment recommendations for patients with mild and moderate renal impairment. 2.6 Dosage Modifications Due to Drug Interactions Rosuvastatin Tablets Dosage Modifications Due to Drug Interactions Table 1 displays dosage modifications for rosuvastatin tablets due to drug interactions [see Warnings and Precautions ( 5.1 ) and Drug Interactions ( 7.1 )]. Table 1: Rosuvastatin Tablets Dosage Modifications Due to Drug Interactions Concomitantly Used Drug Rosuvastatin Tablets Dosage Modifications Cyclosporine Do not exceed 5 mg once daily. Teriflunomide Do not exceed 10 mg once daily. Enasidenib Do not exceed 10 mg once daily. Capmatinib Do not exceed 10 mg once daily. Fostamatinib Do not exceed 20 mg once daily. Febuxostat Do not exceed 20 mg once daily. Gemfibrozil Avoid concomitant use. If used'],
    ['What is the recommended starting dose and maximum dose of amlodipine besylate for adults, including adjustments for elderly or hepatic insufficiency patients?', '2 DOSAGE AND ADMINISTRATION •Adult recommended starting dose: 5 mg once daily with maximum dose 10 mg once daily. ( 2.1 ) о Small, fragile, or elderly patients, or patients with hepatic insufficiency may be started on 2.5 mg once daily. ( 2.1 ) •Pediatric starting dose: 2.5 mg to 5 mg once daily. ( 2.2 ) Important Limitation : Doses in excess of 5 mg daily have not been studied in pediatric patients. ( 2.2 ) 2.1 Adults The usual initial antihypertensive oral dose of amlodipine besylate tablet is 5 mg once daily and the maximum dose is 10 mg once daily. Small, fragile, or elderly patients, or patients with hepatic insufficiency may be started on 2.5 mg once daily and this dose may be used when adding amlodipine besylate tablet to other antihypertensive therapy. Adjust dosage according to blood pressure goals. In general, wait 7 to 14 days between titration steps. Titrate more rapidly, however, if clinically warranted, provided the patient is assessed frequently. Angina The recommended dose for chronic stable or vasospastic angina is 5 to 10 mg, with the lower dose suggested in the elderly and in patients with hepatic insufficiency. Most patients will require 10'],
    ['What are the risks of using flumazenil in benzodiazepine overdose, including seizure risks and contraindications?', 'overdosage, can lead to withdrawal and adverse reactions, including seizures, particularly in the context of mixed overdosage with drugs that increase seizure risk (e.g., tricyclic and tetracyclic antidepressants) and in patients with long-term benzodiazepine use and physical dependency. The risk of withdrawal seizures with flumazenil use may be increased in patients with epilepsy. Flumazenil is contraindicated in patients who have received a benzodiazepine for control of a potentially life-threatening condition (e.g., status epilepticus). If the decision is made to use flumazenil, it should be used as an adjunct to, not as a substitute for, supportive management of benzodiazepine overdosage. See the flumazenil injection Prescribing Information. Consider contacting the Poison Help line (1-800-222-1222) or a medical toxicologist for additional overdosage management recommendations.'],
    ['What pregnancy exposure registry monitors outcomes for women taking clonazepam, and what neonatal risks are reported with late-pregnancy benzodiazepine use?', 'Pregnancy: Pregnancy Exposure Registry There is a pregnancy exposure registry that monitors pregnancy outcomes in women exposed to AEDs, such as clonazepam tablets, during pregnancy. Healthcare providers are encouraged to recommend that pregnant women taking clonazepam tablets enroll in the NAAED Pregnancy Registry by calling 1-888-233-2334 or online at http://www.aedpregnancyregistry.org/. Risk Summary Neonates born to mothers using benzodiazepines late in pregnancy have been reported to experience symptoms of sedation and/or neonatal withdrawal (see WARNINGS: Neonatal Sedation and Withdrawal Syndrome, and Clinical Considerations ). Available data from published observational studies of pregnant women exposed to benzodiazepines do not report a clear association with benzodiazepines and major birth defects (see Data ). Administration of clonazepam to pregnant rabbits during the period of organogenesis resulted in developmental toxicity, including increased incidences of fetal malformations, at doses similar to or below therapeutic doses in patients (see Animal Data). Data for other benzodiazepines suggest the possibility of long-term effects on neurobehavioral and immunological function in animals following prenatal exposure to benzodiazepines at clinically relevant doses. The background risk of major birth defects and miscarriage for the indicated population is unknown. All pregnancies have a background risk of birth defect, loss, or other adverse outcomes. In'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What is the recommended antidote dose for clopidogrel overdose in adult humans?',
    [
        '10 OVERDOSAGE Platelet inhibition by clopidogrel is irreversible and will last for the life of the platelet. Overdose following clopidogrel administration may result in bleeding complications. A single oral dose of clopidogrel at 1500 or 2000 mg/kg was lethal to mice and to rats and at 3000 mg/kg to baboons. Symptoms of acute toxicity were vomiting, prostration, difficult breathing, and gastrointestinal hemorrhage in animals. Based on biological plausibility, platelet transfusion may restore clotting ability.',
        'Precautions ( 5.1 ), Use in Specific Populations ( 8.8 ), and Clinical Pharmacology ( 12.3 )]. 2.5 Recommended Dosage in Patients with Renal Impairment In patients with severe renal impairment (CL cr less than 30 mL/min/1.73 m 2 ) not on hemodialysis, the recommended starting dosage is 5 mg once daily and should not exceed 10 mg once daily [see Warnings and Precautions ( 5.1 ) and Use in Specific Populations ( 8.6 )]. There are no dosage adjustment recommendations for patients with mild and moderate renal impairment. 2.6 Dosage Modifications Due to Drug Interactions Rosuvastatin Tablets Dosage Modifications Due to Drug Interactions Table 1 displays dosage modifications for rosuvastatin tablets due to drug interactions [see Warnings and Precautions ( 5.1 ) and Drug Interactions ( 7.1 )]. Table 1: Rosuvastatin Tablets Dosage Modifications Due to Drug Interactions Concomitantly Used Drug Rosuvastatin Tablets Dosage Modifications Cyclosporine Do not exceed 5 mg once daily. Teriflunomide Do not exceed 10 mg once daily. Enasidenib Do not exceed 10 mg once daily. Capmatinib Do not exceed 10 mg once daily. Fostamatinib Do not exceed 20 mg once daily. Febuxostat Do not exceed 20 mg once daily. Gemfibrozil Avoid concomitant use. If used',
        '2 DOSAGE AND ADMINISTRATION •Adult recommended starting dose: 5 mg once daily with maximum dose 10 mg once daily. ( 2.1 ) о Small, fragile, or elderly patients, or patients with hepatic insufficiency may be started on 2.5 mg once daily. ( 2.1 ) •Pediatric starting dose: 2.5 mg to 5 mg once daily. ( 2.2 ) Important Limitation : Doses in excess of 5 mg daily have not been studied in pediatric patients. ( 2.2 ) 2.1 Adults The usual initial antihypertensive oral dose of amlodipine besylate tablet is 5 mg once daily and the maximum dose is 10 mg once daily. Small, fragile, or elderly patients, or patients with hepatic insufficiency may be started on 2.5 mg once daily and this dose may be used when adding amlodipine besylate tablet to other antihypertensive therapy. Adjust dosage according to blood pressure goals. In general, wait 7 to 14 days between titration steps. Titrate more rapidly, however, if clinically warranted, provided the patient is assessed frequently. Angina The recommended dose for chronic stable or vasospastic angina is 5 to 10 mg, with the lower dose suggested in the elderly and in patients with hepatic insufficiency. Most patients will require 10',
        'overdosage, can lead to withdrawal and adverse reactions, including seizures, particularly in the context of mixed overdosage with drugs that increase seizure risk (e.g., tricyclic and tetracyclic antidepressants) and in patients with long-term benzodiazepine use and physical dependency. The risk of withdrawal seizures with flumazenil use may be increased in patients with epilepsy. Flumazenil is contraindicated in patients who have received a benzodiazepine for control of a potentially life-threatening condition (e.g., status epilepticus). If the decision is made to use flumazenil, it should be used as an adjunct to, not as a substitute for, supportive management of benzodiazepine overdosage. See the flumazenil injection Prescribing Information. Consider contacting the Poison Help line (1-800-222-1222) or a medical toxicologist for additional overdosage management recommendations.',
        'Pregnancy: Pregnancy Exposure Registry There is a pregnancy exposure registry that monitors pregnancy outcomes in women exposed to AEDs, such as clonazepam tablets, during pregnancy. Healthcare providers are encouraged to recommend that pregnant women taking clonazepam tablets enroll in the NAAED Pregnancy Registry by calling 1-888-233-2334 or online at http://www.aedpregnancyregistry.org/. Risk Summary Neonates born to mothers using benzodiazepines late in pregnancy have been reported to experience symptoms of sedation and/or neonatal withdrawal (see WARNINGS: Neonatal Sedation and Withdrawal Syndrome, and Clinical Considerations ). Available data from published observational studies of pregnant women exposed to benzodiazepines do not report a clear association with benzodiazepines and major birth defects (see Data ). Administration of clonazepam to pregnant rabbits during the period of organogenesis resulted in developmental toxicity, including increased incidences of fetal malformations, at doses similar to or below therapeutic doses in patients (see Animal Data). Data for other benzodiazepines suggest the possibility of long-term effects on neurobehavioral and immunological function in animals following prenatal exposure to benzodiazepines at clinically relevant doses. The background risk of major birth defects and miscarriage for the indicated population is unknown. All pregnancies have a background risk of birth defect, loss, or other adverse outcomes. In',
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

* Size: 897 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 897 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                          | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                              | float                                                          |
  | details | <ul><li>min: 41 characters</li><li>mean: 106.97 characters</li><li>max: 256 characters</li></ul> | <ul><li>min: 120 characters</li><li>mean: 1087.05 characters</li><li>max: 1500 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.33</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What is the recommended antidote dose for clopidogrel overdose in adult humans?</code>                                                                                | <code>10 OVERDOSAGE Platelet inhibition by clopidogrel is irreversible and will last for the life of the platelet. Overdose following clopidogrel administration may result in bleeding complications. A single oral dose of clopidogrel at 1500 or 2000 mg/kg was lethal to mice and to rats and at 3000 mg/kg to baboons. Symptoms of acute toxicity were vomiting, prostration, difficult breathing, and gastrointestinal hemorrhage in animals. Based on biological plausibility, platelet transfusion may restore clotting ability.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>0.0</code> |
  | <code>What are the adverse effects of rosuvastatin when combined with cyclosporine in renal impairment?</code>                                                              | <code>Precautions ( 5.1 ), Use in Specific Populations ( 8.8 ), and Clinical Pharmacology ( 12.3 )]. 2.5 Recommended Dosage in Patients with Renal Impairment In patients with severe renal impairment (CL cr less than 30 mL/min/1.73 m 2 ) not on hemodialysis, the recommended starting dosage is 5 mg once daily and should not exceed 10 mg once daily [see Warnings and Precautions ( 5.1 ) and Use in Specific Populations ( 8.6 )]. There are no dosage adjustment recommendations for patients with mild and moderate renal impairment. 2.6 Dosage Modifications Due to Drug Interactions Rosuvastatin Tablets Dosage Modifications Due to Drug Interactions Table 1 displays dosage modifications for rosuvastatin tablets due to drug interactions [see Warnings and Precautions ( 5.1 ) and Drug Interactions ( 7.1 )]. Table 1: Rosuvastatin Tablets Dosage Modifications Due to Drug Interactions Concomitantly Used Drug Rosuvastatin Tablets Dosage Modifications Cyclosporine Do not exceed 5 mg once daily. Teriflunomide Do...</code> | <code>0.0</code> |
  | <code>What is the recommended starting dose and maximum dose of amlodipine besylate for adults, including adjustments for elderly or hepatic insufficiency patients?</code> | <code>2 DOSAGE AND ADMINISTRATION •Adult recommended starting dose: 5 mg once daily with maximum dose 10 mg once daily. ( 2.1 ) о Small, fragile, or elderly patients, or patients with hepatic insufficiency may be started on 2.5 mg once daily. ( 2.1 ) •Pediatric starting dose: 2.5 mg to 5 mg once daily. ( 2.2 ) Important Limitation : Doses in excess of 5 mg daily have not been studied in pediatric patients. ( 2.2 ) 2.1 Adults The usual initial antihypertensive oral dose of amlodipine besylate tablet is 5 mg once daily and the maximum dose is 10 mg once daily. Small, fragile, or elderly patients, or patients with hepatic insufficiency may be started on 2.5 mg once daily and this dose may be used when adding amlodipine besylate tablet to other antihypertensive therapy. Adjust dosage according to blood pressure goals. In general, wait 7 to 14 days between titration steps. Titrate more rapidly, however, if clinically warranted, provided the patient is assessed frequently. Angina The recommended d...</code> | <code>1.0</code> |
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