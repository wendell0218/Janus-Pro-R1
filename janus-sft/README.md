## Supervised Fine-Tuning (Stage 1)

**0. Environment Preparation**

First please clone our repo and prepare the python environment. We recommend using Python>=3.10. 
```bash
git clone https://github.com/wendell0218/Janus-Pro-R1.git
cd Janus-Pro-R1

conda create -n janus-pro-r1-sft python=3.11
conda activate janus-pro-r1-sft
pip install -r requirements-sft.txt
```

**1. Training Configuration**

Before starting the training, you need to prepare a configuration file in advance. We provide an example for reference: `configs/t2i_generation.yml`. This YAML configuration file defines the training settings for SFT. It includes sections for general training setup, optimization strategies, model paths, and data loading. 

To run the training code, you need to specify the following four parameters:

- `output_path`: Path to save model checkpoints and outputs.
- `log_path`: Path to store training logs.
- `model_path`: Path to the pretrained model.
- `processor_path`: Path to the processor.

**2. Prepare Training Data**

We provide an example data sample to clarify the required format for training data. For text-to-image generation, you can refer to ``data/t2i_examples``, and for image editing, you can refer to ``data/editing_examples``.

Specifically, for text-to-image, each data sample should follow the format below:

```json
{
  "promptid": "<promptid>",
  "prompt": "<prompt>",
  "data": [
    {
      "id": "<id>",
      "img_path": "<img_path>",
      "reason": "<reason>",
      "prob": "<prob>",
      "state": "<state>"
    }
  ]
}
```

For editing, each data sample should follow the format below:

```json
{
  "sample_id": "<sample_id>",
  "text": "<text>",
  "input": "<input>",
  "output": "<output>"
}
```

**3. Enjoy Training**

Next, you only need to run a single line of code to start training, try it out now!

```bash
python launch.py --args_yml_fn configs/t2i_generation.yml
```