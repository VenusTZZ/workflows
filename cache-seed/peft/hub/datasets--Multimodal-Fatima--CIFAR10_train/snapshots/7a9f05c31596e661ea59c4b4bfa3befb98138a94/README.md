---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': airplane
          '1': automobile
          '2': bird
          '3': cat
          '4': deer
          '5': dog
          '6': frog
          '7': horse
          '8': ship
          '9': truck
  - name: id
    dtype: int64
  - name: clip_tags_LAION_ViT_H_14_2B_simple_specific
    dtype: string
  - name: clip_tags_LAION_ViT_H_14_2B_ensemble_specific
    dtype: string
  - name: clip_tags_ViT_L_14_simple_specific
    dtype: string
  - name: Attributes_LAION_ViT_H_14_2B_descriptors_text_davinci_003_full
    sequence: string
  - name: Attributes_ViT_L_14_descriptors_text_davinci_003_full
    sequence: string
  splits:
  - name: train
    num_bytes: 140322483.0
    num_examples: 50000
  download_size: 119393875
  dataset_size: 140322483.0
---
# Dataset Card for "CIFAR10_train"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)