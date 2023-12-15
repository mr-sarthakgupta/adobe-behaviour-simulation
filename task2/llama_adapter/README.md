# LLaMA-Adapter: Efficient Fine-tuning of LLaMA 

## Setup

For setting up the environment:

```bash
conda create -n llama_adapter -y python=3.8
conda activate llama_adapter

# install pytorch
conda install pytorch cudatoolkit -c pytorch -y

# install dependency and llama-adapter
pip install -r requirements.txt
pip install -e .
```

## Inference

Please request access to the pre-trained LLaMA from [this form](https://forms.gle/jk851eBVbX1m5TAv5) (official) or download the LLaMA-7B from [Hugging Face](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) (unofficial). Then, obtain the weights of our LLaMA-Adapter from [here](https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.1.0.0/llama_adapter_len10_layer30_release.pth). We denote the path to the downloaded weights of LLaMA and adapters as `TARGET_FOLDER` and `ADAPTER_PATH`.

Here is an example to generate instruction-following sentences with 7B LLaMA model and our LLaMA-Adapter:
```bash
torchrun --nproc_per_node 1 example.py \
         --ckpt_dir $TARGET_FOLDER/model_size\
         --tokenizer_path $TARGET_FOLDER/tokenizer.model \
         --adapter_path $ADAPTER_PATH
```

## Fine-Tuning

```bash
cd alpaca_finetuning_v1

torchrun --nproc_per_node 8 finetuning.py \
         --model Llama7B_adapter \
         --llama_model_path $TARGET_FOLDER/ \
         --data_path $DATA_PATH/alpaca_data.json \
         --adapter_layer 30 \
         --adapter_len 10 \
         --max_seq_len 512 \
         --batch_size 4 \
         --epochs 5 \
         --warmup_epochs 2 \
         --blr 9e-3 \
         --weight_decay 0.02 \
         --output_dir ./checkpoint/
```
