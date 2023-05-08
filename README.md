# Dinov2TinyImageNet

## Self supervised image classification on tinyimagenet and imagenet dataset.

- I make use of the DINO(Distillation of knowledge using No labels) approach for finetuning custom
student teacher network.
- The base model I used were DINOv2base and DINOv2small. Also VITtiny and small

## Running the experiment:
Inorder to run the finetuning/training of the code on custom dataset, make sure to change the dataset in dataset_creation file by giving appropriate path and run the following command.

```python
python main.py --save_path dinocolabmodel --batch_size 64 --num_workers 4 --epochs 3 --save_every 25 --grad_accum_steps 4 --transform_type hard --test_split_percent 0.2 --model_name dinov2 --num_classes 200 --optimizer adamw --scheduler custom --baseline --evaluate
```
