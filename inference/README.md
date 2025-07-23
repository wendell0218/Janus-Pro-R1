
# Inference

We illustrate the inference process of introspective text-to-image generation under the simplest scenario, where the model performs a one-time image self-evaluation and image regeneration after the initial text-to-image generation.

First, the python environment for inference is the same as that for SFT. Specifically, please clone our repo and prepare the python environment. We recommend using Python>=3.10. 
```bash
git clone https://github.com/wendell0218/Janus-Pro-R1.git
cd Janus-Pro-R1

conda create -n janus-pro-r1-sft python=3.11
conda activate janus-pro-r1-sft
pip install -r requirements-sft.txt
```

Then please prepare the model Janus-Pro-R1-7B, which utilizes Janus-Pro-7B as the backbone model. You can download the corresponding model from [🤗https://huggingface.co/midbee/Janus-Pro-R1-7B](https://huggingface.co/midbee/Janus-Pro-R1-7B).

You can conduct the inference process using the following command. ``model_path`` refers to the local path where you have downloaded Janus-Pro-R1-7B.

```bash
  python inference/inference.py \
      --model_path $CKPT_PATH \
      --caption "a brown giraffe and a white stop sign" \
      --gen_path "results/samples" \
      --reason_path "results/reason.jsonl" \
      --regen_path "results/regen_samples" \
      --cfg 5.0 \
      --parallel_size 4
  ```
After completing the inference, the structure of the `results` directory will be as follows:
  
  ```text
  results/
  ├── reason.jsonl
  ├── samples/
  │   ├── 0000.png
  │   ├── 0001.png
  │   ├── 0002.png
  │   └── 0003.png
  └── regen_samples/
      ├── 0000.png
      ├── 0001.png
      ├── 0002.png
      └── 0003.png
  ```


Also, to facilitate the reader's understanding, we also split the inference logic of introspective text-to-image generation into three separate files to demonstrate each sub-process.

- ### Text-to-Image Generation

  You can generate images from text prompts using the following command. 

  ```bash
  python inference/t2i.py \
      --model_path $CKPT_PATH \
      --caption "a brown giraffe and a white stop sign" \
      --gen_path "results/samples" \
      --cfg 5.0 \
      --parallel_size 4
  ```

  After running the above command, the generated images will be saved under the directory specified by `gen_path`. For example, if `gen_path` is set to `results/samples` and `parallel_size` is 4, the output directory will look like this:

  ```text
  results/
  └── samples/
      ├── 0000.png
      ├── 0001.png
      ├── 0002.png
      └── 0003.png
  ```

- ### Self-evaluation on Image-Text Consistency

  Before self-evaluating image-text consistency, ensure that `gen_path` points to a directory with the same structure in the **Text-to-Image Generation**.

  ```bash
  python inference/reflect.py \
      --model_path $CKPT_PATH \
      --caption "a brown giraffe and a white stop sign" \
      --gen_path "results/samples" \
      --reason_path "results/reason.jsonl"
  ```

- ### Image Regeneration Enhanced by Reflection

  You can then regenerate higher-quality images based on the initial generation and corresponding reflection using the following command:

  ```bash
  python inference/regen.py \
      --model_path $CKPT_PATH \
      --caption "a brown giraffe and a white stop sign" \
      --gen_path "results/samples" \
      --reason_path "results/reason.jsonl" \
      --regen_path "results/regen_samples" \
      --cfg 5.0 
  ```
