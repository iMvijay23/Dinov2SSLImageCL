## Finetuned DINO model on tiny imagenet dataset.

### Download the model at this [link](https://drive.google.com/file/d/1-5hhFMxgxjhVfs4Y-aLoYR1dWxKnU8eT/view?usp=sharing)

Run inference on this model by setting the evaluate argument to `true`

eg. 
```python 
!python main.py --save_path dinocolabmodel --batch_size 64 --num_workers 4 --epochs 3 --save_every 25 --grad_accum_steps 4 --transform_type hard --test_split_percent 0.2 --model_name dinov2 --num_classes 200 --optimizer adam --scheduler cosineannealing --baseline --evaluate
```
