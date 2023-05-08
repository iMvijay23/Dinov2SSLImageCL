# Finetunig DINOv2 for Unsupervised Image Classification

## Self supervised image classification on tinyimagenet and imagenet dataset.

- Repository contains the implementation of DINO(Distillation of knowledge using No labels) approach for finetuning custom student teacher training network.
- The base models used are DINOv2base and DINOv2small. Also experimented with VIT tiny and small.

## Running the experiment:
In order to run the finetuning/training of the model on custom dataset, make sure to change the dataset in dataset_preperation file by giving appropriate path and run the following command.

```python
python main.py --save_path path/to/save --batch_size 64 --num_workers 4 --epochs 3 --save_every 25 --grad_accum_steps 4 --transform_type hard --test_split_percent 0.2 --model_name dinov2 --num_classes 200 --optimizer adamw --scheduler custom --baseline --evaluate
```


## Acknowledgment for references
[DINO](https://github.com/facebookresearch/dino)
[DINOv2](https://github.com/facebookresearch/dinov2)
