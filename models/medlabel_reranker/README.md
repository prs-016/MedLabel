---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:100
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
    ['What specific changes in INR occur when patients take warfarin together with rifampin?', 'telithromycin, tipranavir, voriconazole, zileuton armodafinil, amprenavir, aprepitant, bosentan, carbamazepine, efavirenz, etravirine, modafinil, nafcillin, phenytoin, pioglitazone, prednisone, rifampin, rufinamide 7.3 Drugs that Increase Bleeding Risk Examples of drugs known to increase the risk of bleeding are presented in Table 3 . Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin. Table 3: Drugs that Can Increase the Risk of Bleeding Drug Class Specific Drugs Anticoagulants argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin Antiplatelet Agents aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine Non-steroidal Anti-Inflammatory Agents celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac Serotonin Reuptake Inhibitors citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, vilazodone 7.4 Antibiotics and Antifungals There have been reports of changes in INR in patients taking warfarin and antibiotics or antifungals, but clinical pharmacokinetic studies have not shown consistent effects of these agents on plasma concentrations of warfarin. Closely monitor INR when starting or stopping any antibiotic or antifungal in patients taking warfarin. 7.5 Botanical (Herbal) Products and Foods More frequent INR monitoring should be performed when starting or stopping botanicals. Few adequate, well-controlled studies evaluating the potential for'],
    ['What specific numerical values define the therapeutic INR range for warfarin therapy?', 'used in this table. Other co-inherited VKORC1 variants may also be important determinants of warfarin dose. 2.4 Monitoring to Achieve Optimal Anticoagulation Warfarin sodium tablets have a narrow therapeutic range (index), and their action may be affected by factors such as other drugs and dietary vitamin K. Therefore, anticoagulation must be carefully monitored during warfarin sodium tablets therapy. Determine the INR daily after the administration of the initial dose until INR results stabilize in the therapeutic range. After stabilization, maintain dosing within the therapeutic range by performing periodic INRs. The frequency of performing INR should be based on the clinical situation but generally acceptable intervals for INR determinations are 1 to 4 weeks. Perform additional INR tests when other warfarin products are interchanged with warfarin sodium tablets, as well as whenever other medications are initiated, discontinued, or taken irregularly. Heparin, a common concomitant drug, increases the INR [see Dosage and Administration ( 2.8 ) and Drug Interactions ( 7 )] . Determinations of whole blood clotting and bleeding times are not effective measures for monitoring of warfarin sodium tablets therapy. 2.5 Renal Impairment No dosage adjustment is necessary for patients with renal failure. Monitor INR more frequently in patients with'],
    ['Which specific antibiotics inhibit CYP3A4 and increase bleeding risk with warfarin sodium?', '7 DRUG INTERACTIONS Concomitant use of drugs that increase bleeding risk, antibiotics, antifungals, botanical (herbal) products, and inhibitors and inducers of CYP2C9, 1A2, or 3A4. ( 7 ) Consult labeling of all concurrently used drugs for complete information about interactions with warfarin sodium or increased risks for bleeding. ( 7 ) 7.1 General Information Drugs may interact with warfarin sodium through pharmacodynamic or pharmacokinetic mechanisms. Pharmacodynamic mechanisms for drug interactions with warfarin sodium are synergism (impaired hemostasis, reduced clotting factor synthesis), competitive antagonism (vitamin K), and alteration of the physiologic control loop for vitamin K metabolism (hereditary resistance). Pharmacokinetic mechanisms for drug interactions with warfarin sodium are mainly enzyme induction, enzyme inhibition, and reduced plasma protein binding. It is important to note that some drugs may interact by more than one mechanism. More frequent INR monitoring should be performed when starting or stopping other drugs, including botanicals, or when changing dosages of other drugs, including drugs intended for short-term use (e.g., antibiotics, antifungals, corticosteroids) [ see Boxed Warning ]. Consult the labeling of all concurrently used drugs to obtain further information about interactions with warfarin sodium or adverse reactions pertaining to bleeding. 7.2 CYP450 Interactions CYP450 isozymes involved in the'],
    ['What citation number is given for the mention of bleeding?', 'of bleeding. ( 17 )'],
    ['What specific numerical maintenance dose ranges does Table 1 list for each CYP2C9 and VKORC1 genotype combination?', 'formation. Individualize the duration of therapy for each patient. In general, anticoagulant therapy should be continued until the danger of thrombosis and embolism has passed [see Dosage and Administration ( 2.2 )] . Dosing Recommendations without Consideration of Genotype If the patient’s CYP2C9 and VKORC1 genotypes are not known, the initial dose of warfarin sodium tablets is usually 2 to 5 mg once daily. Determine each patient’s dosing needs by close monitoring of the INR response and consideration of the indication being treated. Typical maintenance doses are 2 to 10 mg once daily. Dosing Recommendations with Consideration of Genotype Table 1 displays three ranges of expected maintenance warfarin sodium tablets doses observed in subgroups of patients having different combinations of CYP2C9 and VKORC1 gene variants [see Clinical Pharmacology ( 12.5 )] . If the patient’s CYP2C9 and/or VKORC1 genotype are known, consider these ranges in choosing the initial dose. Patients with CYP2C9 *1/*3, *2/*2, *2/*3, and *3/*3 may require more prolonged time (> 2 to 4 weeks) to achieve maximum INR effect for a given dosage regimen than patients without these CYP variants. Table 1: Three Ranges of Expected Maintenance Warfarin Sodium Tablets Daily Doses Based on CYP2C9 and VKORC1'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What specific changes in INR occur when patients take warfarin together with rifampin?',
    [
        'telithromycin, tipranavir, voriconazole, zileuton armodafinil, amprenavir, aprepitant, bosentan, carbamazepine, efavirenz, etravirine, modafinil, nafcillin, phenytoin, pioglitazone, prednisone, rifampin, rufinamide 7.3 Drugs that Increase Bleeding Risk Examples of drugs known to increase the risk of bleeding are presented in Table 3 . Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin. Table 3: Drugs that Can Increase the Risk of Bleeding Drug Class Specific Drugs Anticoagulants argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin Antiplatelet Agents aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine Non-steroidal Anti-Inflammatory Agents celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac Serotonin Reuptake Inhibitors citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, vilazodone 7.4 Antibiotics and Antifungals There have been reports of changes in INR in patients taking warfarin and antibiotics or antifungals, but clinical pharmacokinetic studies have not shown consistent effects of these agents on plasma concentrations of warfarin. Closely monitor INR when starting or stopping any antibiotic or antifungal in patients taking warfarin. 7.5 Botanical (Herbal) Products and Foods More frequent INR monitoring should be performed when starting or stopping botanicals. Few adequate, well-controlled studies evaluating the potential for',
        'used in this table. Other co-inherited VKORC1 variants may also be important determinants of warfarin dose. 2.4 Monitoring to Achieve Optimal Anticoagulation Warfarin sodium tablets have a narrow therapeutic range (index), and their action may be affected by factors such as other drugs and dietary vitamin K. Therefore, anticoagulation must be carefully monitored during warfarin sodium tablets therapy. Determine the INR daily after the administration of the initial dose until INR results stabilize in the therapeutic range. After stabilization, maintain dosing within the therapeutic range by performing periodic INRs. The frequency of performing INR should be based on the clinical situation but generally acceptable intervals for INR determinations are 1 to 4 weeks. Perform additional INR tests when other warfarin products are interchanged with warfarin sodium tablets, as well as whenever other medications are initiated, discontinued, or taken irregularly. Heparin, a common concomitant drug, increases the INR [see Dosage and Administration ( 2.8 ) and Drug Interactions ( 7 )] . Determinations of whole blood clotting and bleeding times are not effective measures for monitoring of warfarin sodium tablets therapy. 2.5 Renal Impairment No dosage adjustment is necessary for patients with renal failure. Monitor INR more frequently in patients with',
        '7 DRUG INTERACTIONS Concomitant use of drugs that increase bleeding risk, antibiotics, antifungals, botanical (herbal) products, and inhibitors and inducers of CYP2C9, 1A2, or 3A4. ( 7 ) Consult labeling of all concurrently used drugs for complete information about interactions with warfarin sodium or increased risks for bleeding. ( 7 ) 7.1 General Information Drugs may interact with warfarin sodium through pharmacodynamic or pharmacokinetic mechanisms. Pharmacodynamic mechanisms for drug interactions with warfarin sodium are synergism (impaired hemostasis, reduced clotting factor synthesis), competitive antagonism (vitamin K), and alteration of the physiologic control loop for vitamin K metabolism (hereditary resistance). Pharmacokinetic mechanisms for drug interactions with warfarin sodium are mainly enzyme induction, enzyme inhibition, and reduced plasma protein binding. It is important to note that some drugs may interact by more than one mechanism. More frequent INR monitoring should be performed when starting or stopping other drugs, including botanicals, or when changing dosages of other drugs, including drugs intended for short-term use (e.g., antibiotics, antifungals, corticosteroids) [ see Boxed Warning ]. Consult the labeling of all concurrently used drugs to obtain further information about interactions with warfarin sodium or adverse reactions pertaining to bleeding. 7.2 CYP450 Interactions CYP450 isozymes involved in the',
        'of bleeding. ( 17 )',
        'formation. Individualize the duration of therapy for each patient. In general, anticoagulant therapy should be continued until the danger of thrombosis and embolism has passed [see Dosage and Administration ( 2.2 )] . Dosing Recommendations without Consideration of Genotype If the patient’s CYP2C9 and VKORC1 genotypes are not known, the initial dose of warfarin sodium tablets is usually 2 to 5 mg once daily. Determine each patient’s dosing needs by close monitoring of the INR response and consideration of the indication being treated. Typical maintenance doses are 2 to 10 mg once daily. Dosing Recommendations with Consideration of Genotype Table 1 displays three ranges of expected maintenance warfarin sodium tablets doses observed in subgroups of patients having different combinations of CYP2C9 and VKORC1 gene variants [see Clinical Pharmacology ( 12.5 )] . If the patient’s CYP2C9 and/or VKORC1 genotype are known, consider these ranges in choosing the initial dose. Patients with CYP2C9 *1/*3, *2/*2, *2/*3, and *3/*3 may require more prolonged time (> 2 to 4 weeks) to achieve maximum INR effect for a given dosage regimen than patients without these CYP variants. Table 1: Three Ranges of Expected Maintenance Warfarin Sodium Tablets Daily Doses Based on CYP2C9 and VKORC1',
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

* Size: 100 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                         | label                                                         |
  |:--------|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                             | float                                                         |
  | details | <ul><li>min: 49 characters</li><li>mean: 102.22 characters</li><li>max: 204 characters</li></ul> | <ul><li>min: 15 characters</li><li>mean: 1182.08 characters</li><li>max: 1945 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:--------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What specific changes in INR occur when patients take warfarin together with rifampin?</code>     | <code>telithromycin, tipranavir, voriconazole, zileuton armodafinil, amprenavir, aprepitant, bosentan, carbamazepine, efavirenz, etravirine, modafinil, nafcillin, phenytoin, pioglitazone, prednisone, rifampin, rufinamide 7.3 Drugs that Increase Bleeding Risk Examples of drugs known to increase the risk of bleeding are presented in Table 3 . Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin. Table 3: Drugs that Can Increase the Risk of Bleeding Drug Class Specific Drugs Anticoagulants argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin Antiplatelet Agents aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine Non-steroidal Anti-Inflammatory Agents celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac Serotonin Reuptake Inhibitors citalopram, desvenlafaxine, duloxet...</code> | <code>0.0</code> |
  | <code>What specific numerical values define the therapeutic INR range for warfarin therapy?</code>      | <code>used in this table. Other co-inherited VKORC1 variants may also be important determinants of warfarin dose. 2.4 Monitoring to Achieve Optimal Anticoagulation Warfarin sodium tablets have a narrow therapeutic range (index), and their action may be affected by factors such as other drugs and dietary vitamin K. Therefore, anticoagulation must be carefully monitored during warfarin sodium tablets therapy. Determine the INR daily after the administration of the initial dose until INR results stabilize in the therapeutic range. After stabilization, maintain dosing within the therapeutic range by performing periodic INRs. The frequency of performing INR should be based on the clinical situation but generally acceptable intervals for INR determinations are 1 to 4 weeks. Perform additional INR tests when other warfarin products are interchanged with warfarin sodium tablets, as well as whenever other medications are initiated, discontinued, or taken irregularly. Heparin, a common concomitant dru...</code> | <code>0.0</code> |
  | <code>Which specific antibiotics inhibit CYP3A4 and increase bleeding risk with warfarin sodium?</code> | <code>7 DRUG INTERACTIONS Concomitant use of drugs that increase bleeding risk, antibiotics, antifungals, botanical (herbal) products, and inhibitors and inducers of CYP2C9, 1A2, or 3A4. ( 7 ) Consult labeling of all concurrently used drugs for complete information about interactions with warfarin sodium or increased risks for bleeding. ( 7 ) 7.1 General Information Drugs may interact with warfarin sodium through pharmacodynamic or pharmacokinetic mechanisms. Pharmacodynamic mechanisms for drug interactions with warfarin sodium are synergism (impaired hemostasis, reduced clotting factor synthesis), competitive antagonism (vitamin K), and alteration of the physiologic control loop for vitamin K metabolism (hereditary resistance). Pharmacokinetic mechanisms for drug interactions with warfarin sodium are mainly enzyme induction, enzyme inhibition, and reduced plasma protein binding. It is important to note that some drugs may interact by more than one mechanism. More frequent INR monitoring sho...</code> | <code>0.0</code> |
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