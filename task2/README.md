## Task 2

### Training Scripts and Inference

#### 1. Initial Captioning: Blip Inference 
Initial inference of images, videos and gif files for generating captions and prompts to give to LLM based models for finetuning and generating quality captions. Use this to generate data for `finetuning.json` file to be stored in the data folder.

```
>> cd ./task2/blip-inference/
>> git clone https://github.com/salesforce/BLIP
>> cd ../..
>> python3 ./task2/blip-inference/blip-inference-data.py --data-path=./data/complete_username_data.xlsx --username-data=./data/username_data.csv
```

Note: 
Install these libraries like this to use bits and bytes and other parts used in file. Also for this task try to use a system with a single GPU memory of atleast 10 GB and RAM of about 16 GB. We used Kaggle free GB notebooks for running these tasks on both test and train dataset.
```
>> pip install -i https://test.pypi.org/simple/ bitsandbytes
>> pip install flash-attn --no-build-isolation
```

#### 2. Post Processing
Here we will try to use LLM based approaches to post process the captions generated by the BLIP model (1 and 2) to generate prompts and finetuning data that can be directly used to finetune and infer the new LLM models like LLAMA, FastGPT and others to generate better and proper captions for the Twitter posts.


##### (A) LLAMA Adapter


##### (B) Lora Fintuning


##### (C) Incontext Learning
We used the LLAMA model tokenizer, finetuned it on the dataset and then infered using the LLAMA finetuned weights as in LLAMA adapter and then used incontext learning pipline after that to get the 


##### (D) Fast-GPT
We used Fast-GPT training and eval scripts to get quick inference results from the training from pytorch's [Fast-GPT](https://github.com/pytorch-labs/gpt-fast#BSD-3-Clause-1-ov-file) licensed by Meta under BSD clase Licence. 
The repo `task2/fast_GPT` is a clone of the PyTorch repo nd we wrote an eval script for understanding how to use the script for the finetuning and later inference of the model. If more work is done on this area, we may significantly reduce the training and the inference 
