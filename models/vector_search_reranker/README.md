---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:449
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
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
    ['Does gabapentin interfere with the metabolism of commonly coadministered antiepileptic drugs?', '7.2 Other Antiepileptic Drugs Gabapentin is not appreciably metabolized nor does it interfere with the metabolism of commonly coadministered antiepileptic drugs [ see Clinical Pharmacology (12.3) ] .'],
    ['What is the mechanism of action of thyroid hormones and how is oral levothyroxine absorbed and bioavailable?', '12 CLINICAL PHARMACOLOGY 12.1 Mechanism of Action Thyroid hormones exert their physiologic actions through control of DNA transcription and protein synthesis. Triiodothyronine (T3) and L-thyroxine (T4) diffuse into the cell nucleus and bind to thyroid receptor proteins attached to DNA. This hormone nuclear receptor complex activates gene transcription and synthesis of messenger RNA and cytoplasmic proteins. The physiological actions of thyroid hormones are produced predominantly by T3, the majority of which (approximately 80%) is derived from T4 by deiodination in peripheral tissues. 12.2 Pharmacodynamics Oral levothyroxine sodium is a synthetic T4 hormone that exerts the same physiologic effect as endogenous T4, thereby maintaining normal T4 levels when a deficiency is present. 12.3 Pharmacokinetics Absorption Absorption of orally administered T4 from the gastrointestinal tract ranges from 40% to 80%. The majority of the levothyroxine sodium dose is absorbed from the jejunum and upper ileum. The relative bioavailability of levothyroxine sodium tablets, compared to an equal nominal dose of oral levothyroxine sodium solution, is approximately 93%. T4 absorption is increased by fasting, and decreased in malabsorption syndromes and by certain foods such as soybeans. Dietary fiber decreases bioavailability of T4. Absorption may also decrease with age. In addition, many drugs and foods affect T4 absorption [see Drug Interactions (7) ] . Distribution Circulating thyroid hormones ar'],
    ['What are the pregnancy risks of tramadol hydrochloride extended-release tablets including effects seen in animal studies?', '8 USE IN SPECIFIC POPULATIONS Pregnancy: May cause fetal harm. ( 8.1 ) Lactation: Breastfeeding not recommended. ( 8.2 ) Severe Hepatic or Renal Impairment: Use not recommended. ( 8.6 , 8.7 ) 8.1 Pregnancy Risk Summary Use of opioid analgesics for an extended period of time during pregnancy may cause neonatal opioid withdrawal syndrome [see Warnings and Precautions (5.4)] . Available data with tramadol hydrochloride extended-release tablets in pregnant women are insufficient to inform a drug-associated risk for major birth defects and miscarriage. In animal reproduction studies, tramadol administration during organogenesis decreased fetal weights and reduced ossification in mice, rats, and rabbits at 1.4, 0.6, and 3.6 times the maximum recommended human daily dosage (MRHD). Tramadol decreased pup body weight and increased pup mortality at 1.2 and 1.9 times the MRHD [see Data]. Based on animal data, advise pregnant women of the potential risk to a fetus. The estimated background risk of major birth defects and miscarriage for the indicated population is unknown. All pregnancies have a background risk of birth defect, loss, or other adverse outcomes. In the U.S. general population, the estimated background risk of major birth defects and miscarriage in clinically recognized pregnancies is 2 to 4% and 15 to 20%, respectively. Clinical Considerations Fetal/Neonatal Adverse Reactions Use of opioid analgesics for an extended period of time during pregnancy for medical or nonmedical'],
    ['How does atorvastatin lower cholesterol by affecting HMG-CoA reductase and LDL receptors?', '12.1 Mechanism of Action Atorvastatin is a selective, competitive inhibitor of HMG-CoA reductase, the rate-limiting enzyme that converts 3-hydroxy-3-\xadmethylglutaryl-coenzyme A to mevalonate, a precursor of sterols, including cholesterol. In animal models, atorvastatin calcium lowers plasma cholesterol and lipoprotein levels by inhibiting HMG-CoA reductase and cholesterol synthesis in the liver and by increasing the number of hepatic LDL receptors on the cell surface to enhance uptake and catabolism of LDL; atorvastatin calcium also reduces LDL production and the number of LDL particles.'],
    ['What are the long-term clinical outcomes of amlodipine binding to nondihydropyridine sites in patients with exertional angina?', '12.1 Mechanism of Action Amlodipine is a dihydropyridine calcium antagonist (calcium ion antagonist or slow-channel blocker) that inhibits the transmembrane influx of calcium ions into vascular smooth muscle and cardiac muscle. Experimental data suggest that amlodipine binds to both dihydropyridine and nondihydropyridine binding sites. The contractile processes of cardiac muscle and vascular smooth muscle are dependent upon the movement of extracellular calcium ions into these cells through specific ion channels. Amlodipine inhibits calcium ion influx across cell membranes selectively, with a greater effect on vascular smooth muscle cells than on cardiac muscle cells. Negative inotropic effects can be detected in vitro but such effects have not been seen in intact animals at therapeutic doses. Serum calcium concentration is not affected by amlodipine. Within the physiologic pH range, amlodipine is an ionized compound (pKa=8.6), and its kinetic interaction with the calcium channel receptor is characterized by a gradual rate of association and dissociation with the receptor binding site, resulting in a gradual onset of effect. Amlodipine is a peripheral arterial vasodilator that acts directly on vascular smooth muscle to cause a reduction in peripheral vascular resistance and reduction in blood pressure. The precise mechanisms by which amlodipine relieves angina have not been fully delineated, but are thought to include the following: Exertional Angina In patients with exertion'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Does gabapentin interfere with the metabolism of commonly coadministered antiepileptic drugs?',
    [
        '7.2 Other Antiepileptic Drugs Gabapentin is not appreciably metabolized nor does it interfere with the metabolism of commonly coadministered antiepileptic drugs [ see Clinical Pharmacology (12.3) ] .',
        '12 CLINICAL PHARMACOLOGY 12.1 Mechanism of Action Thyroid hormones exert their physiologic actions through control of DNA transcription and protein synthesis. Triiodothyronine (T3) and L-thyroxine (T4) diffuse into the cell nucleus and bind to thyroid receptor proteins attached to DNA. This hormone nuclear receptor complex activates gene transcription and synthesis of messenger RNA and cytoplasmic proteins. The physiological actions of thyroid hormones are produced predominantly by T3, the majority of which (approximately 80%) is derived from T4 by deiodination in peripheral tissues. 12.2 Pharmacodynamics Oral levothyroxine sodium is a synthetic T4 hormone that exerts the same physiologic effect as endogenous T4, thereby maintaining normal T4 levels when a deficiency is present. 12.3 Pharmacokinetics Absorption Absorption of orally administered T4 from the gastrointestinal tract ranges from 40% to 80%. The majority of the levothyroxine sodium dose is absorbed from the jejunum and upper ileum. The relative bioavailability of levothyroxine sodium tablets, compared to an equal nominal dose of oral levothyroxine sodium solution, is approximately 93%. T4 absorption is increased by fasting, and decreased in malabsorption syndromes and by certain foods such as soybeans. Dietary fiber decreases bioavailability of T4. Absorption may also decrease with age. In addition, many drugs and foods affect T4 absorption [see Drug Interactions (7) ] . Distribution Circulating thyroid hormones ar',
        '8 USE IN SPECIFIC POPULATIONS Pregnancy: May cause fetal harm. ( 8.1 ) Lactation: Breastfeeding not recommended. ( 8.2 ) Severe Hepatic or Renal Impairment: Use not recommended. ( 8.6 , 8.7 ) 8.1 Pregnancy Risk Summary Use of opioid analgesics for an extended period of time during pregnancy may cause neonatal opioid withdrawal syndrome [see Warnings and Precautions (5.4)] . Available data with tramadol hydrochloride extended-release tablets in pregnant women are insufficient to inform a drug-associated risk for major birth defects and miscarriage. In animal reproduction studies, tramadol administration during organogenesis decreased fetal weights and reduced ossification in mice, rats, and rabbits at 1.4, 0.6, and 3.6 times the maximum recommended human daily dosage (MRHD). Tramadol decreased pup body weight and increased pup mortality at 1.2 and 1.9 times the MRHD [see Data]. Based on animal data, advise pregnant women of the potential risk to a fetus. The estimated background risk of major birth defects and miscarriage for the indicated population is unknown. All pregnancies have a background risk of birth defect, loss, or other adverse outcomes. In the U.S. general population, the estimated background risk of major birth defects and miscarriage in clinically recognized pregnancies is 2 to 4% and 15 to 20%, respectively. Clinical Considerations Fetal/Neonatal Adverse Reactions Use of opioid analgesics for an extended period of time during pregnancy for medical or nonmedical',
        '12.1 Mechanism of Action Atorvastatin is a selective, competitive inhibitor of HMG-CoA reductase, the rate-limiting enzyme that converts 3-hydroxy-3-\xadmethylglutaryl-coenzyme A to mevalonate, a precursor of sterols, including cholesterol. In animal models, atorvastatin calcium lowers plasma cholesterol and lipoprotein levels by inhibiting HMG-CoA reductase and cholesterol synthesis in the liver and by increasing the number of hepatic LDL receptors on the cell surface to enhance uptake and catabolism of LDL; atorvastatin calcium also reduces LDL production and the number of LDL particles.',
        '12.1 Mechanism of Action Amlodipine is a dihydropyridine calcium antagonist (calcium ion antagonist or slow-channel blocker) that inhibits the transmembrane influx of calcium ions into vascular smooth muscle and cardiac muscle. Experimental data suggest that amlodipine binds to both dihydropyridine and nondihydropyridine binding sites. The contractile processes of cardiac muscle and vascular smooth muscle are dependent upon the movement of extracellular calcium ions into these cells through specific ion channels. Amlodipine inhibits calcium ion influx across cell membranes selectively, with a greater effect on vascular smooth muscle cells than on cardiac muscle cells. Negative inotropic effects can be detected in vitro but such effects have not been seen in intact animals at therapeutic doses. Serum calcium concentration is not affected by amlodipine. Within the physiologic pH range, amlodipine is an ionized compound (pKa=8.6), and its kinetic interaction with the calcium channel receptor is characterized by a gradual rate of association and dissociation with the receptor binding site, resulting in a gradual onset of effect. Amlodipine is a peripheral arterial vasodilator that acts directly on vascular smooth muscle to cause a reduction in peripheral vascular resistance and reduction in blood pressure. The precise mechanisms by which amlodipine relieves angina have not been fully delineated, but are thought to include the following: Exertional Angina In patients with exertion',
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

* Size: 449 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 449 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                          | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                              | float                                                          |
  | details | <ul><li>min: 45 characters</li><li>mean: 105.57 characters</li><li>max: 184 characters</li></ul> | <ul><li>min: 130 characters</li><li>mean: 1255.55 characters</li><li>max: 1500 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.33</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Does gabapentin interfere with the metabolism of commonly coadministered antiepileptic drugs?</code>                             | <code>7.2 Other Antiepileptic Drugs Gabapentin is not appreciably metabolized nor does it interfere with the metabolism of commonly coadministered antiepileptic drugs [ see Clinical Pharmacology (12.3) ] .</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <code>1.0</code> |
  | <code>What is the mechanism of action of thyroid hormones and how is oral levothyroxine absorbed and bioavailable?</code>              | <code>12 CLINICAL PHARMACOLOGY 12.1 Mechanism of Action Thyroid hormones exert their physiologic actions through control of DNA transcription and protein synthesis. Triiodothyronine (T3) and L-thyroxine (T4) diffuse into the cell nucleus and bind to thyroid receptor proteins attached to DNA. This hormone nuclear receptor complex activates gene transcription and synthesis of messenger RNA and cytoplasmic proteins. The physiological actions of thyroid hormones are produced predominantly by T3, the majority of which (approximately 80%) is derived from T4 by deiodination in peripheral tissues. 12.2 Pharmacodynamics Oral levothyroxine sodium is a synthetic T4 hormone that exerts the same physiologic effect as endogenous T4, thereby maintaining normal T4 levels when a deficiency is present. 12.3 Pharmacokinetics Absorption Absorption of orally administered T4 from the gastrointestinal tract ranges from 40% to 80%. The majority of the levothyroxine sodium dose is absorbed from the jejunum and upper...</code> | <code>1.0</code> |
  | <code>What are the pregnancy risks of tramadol hydrochloride extended-release tablets including effects seen in animal studies?</code> | <code>8 USE IN SPECIFIC POPULATIONS Pregnancy: May cause fetal harm. ( 8.1 ) Lactation: Breastfeeding not recommended. ( 8.2 ) Severe Hepatic or Renal Impairment: Use not recommended. ( 8.6 , 8.7 ) 8.1 Pregnancy Risk Summary Use of opioid analgesics for an extended period of time during pregnancy may cause neonatal opioid withdrawal syndrome [see Warnings and Precautions (5.4)] . Available data with tramadol hydrochloride extended-release tablets in pregnant women are insufficient to inform a drug-associated risk for major birth defects and miscarriage. In animal reproduction studies, tramadol administration during organogenesis decreased fetal weights and reduced ossification in mice, rats, and rabbits at 1.4, 0.6, and 3.6 times the maximum recommended human daily dosage (MRHD). Tramadol decreased pup body weight and increased pup mortality at 1.2 and 1.9 times the MRHD [see Data]. Based on animal data, advise pregnant women of the potential risk to a fetus. The estimated background risk of...</code> | <code>1.0</code> |
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