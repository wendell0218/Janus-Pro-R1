## Reinforcement Learning (Stage 2)

First please clone our repo and prepare the python environment. We recommend using Python>=3.10. 
```bash
git clone https://github.com/wendell0218/Janus-Pro-R1.git
cd Janus-Pro-R1

conda create -n janus-pro-r1-rl python=3.11
conda activate janus-pro-r1-rl
pip install -r requirements-rl.txt
```

Then, below is the organized file structure with brief descriptions:

```text
janus-rl/
├── recipes/                        # The cases for preparing training data
├── recipes/                        # Configuration files related to GRPO-based training
├── src/
    └── open_r1                     # Source code for the reinforcement learning
        ├── internvl/               # Inference code related to InternVL
        ├── models/                 # Model architecture of Janus-Pro-R1
        ├── configs.py              # Configuration settings for GRPO
        ├── grpo_trainer_t2iv1.py   # Implementation of GRPO (version1) for introspective text-to-image generation
        ├── grpo_trainer_t2iv2.py   # Implementation of GRPO (version2) for introspective text-to-image generation
        ├── grpo_trainer_t2iv3.py   # Implementation of GRPO (version3) for introspective text-to-image generation
        ├── grpo_trainer_editing.py # Implementation of GRPO  for image editing
        ├── grpo_t2i.py             # Entry point for running GRPO for introspective text-to-image generation
        ├── grpo_editing.py         # Entry point for running GRPO for image editing
        ├── internvl_img.py         # Script for reward calculation with InternVL for text-to-image generation
        ├── internvl_edit.py        # Script for reward calculation with InternVL for image editing
        ├── mydataset.py            # Dataset construction
        └── llama.py                # Inference script for Janus-Pro-R1
```

Next, it is necessary to download some essential backbone models as the policy/reference model, as well as VLMs that serve as the reward models.
Specifically, you can download [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) as the backbone model, filling in the local path of the Janus-Pro-7B model weights in the necessary code ([recipes/t2i_generation/grpo.yml#L3 & L45](recipes/t2i_generation/grpo.yml#L3)). Also, be sure to fill in the path to the model checkpoint after SFT in the [recipes/t2i_generation/grpo.yml#L46 & L47](recipes/t2i_generation/grpo.yml#L46). The same applies to the reinforcement learning for image editing.

While for rewars models, there are two optinal ways to call them: online and offline.
For models that are deployed on GPUs with larger memory, such as [InternVL2.5-26B](https://huggingface.co/OpenGVLab/InternVL2_5-26B), online deployment can be adopted, and a URL can be exposed as the calling interface.
For models that are deployed on GPUs with relatively smaller memory, such as [InternVL2.5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B), offline deployment can be used, where the model is directly deployed on the GPUs for training.
This eliminates the need for network communication and provides greater training stability.
Specifically, for the former, after model deployment is complete, the URL path needs to be modified in the corresponding configuration, e.g., [src/open_r1/grpo_t2i.py#L94](src/open_r1/grpo_t2i.py#L94). 
For the latter, the local weight path of the reward model should be filled in [src/open_r1/internvl_img.py#L98](src/open_r1/internvl_img.py#L98).
Moreover, the hyperparameter ``internvl_tp`` in the [configuration file](recipes/t2i_generation/grpo.yml#L61) is used to control the parameter weights of the reward model. A value of ``26b`` corresponds to the reward model of InternVL2.5-26B, indicating an offline deployment; a value of ``8b`` corresponds to the reward model of InternVL2.5-8B, indicating an online deployment. 
**The same applies to image editing.**
Of course, **we strongly recommend using the 26B model as the reward model, especially for image-editing training.**


Finally, after preparing some training data-related files, You can perform RL for **Introspective Text-to-Image Generation** using the following command:

```bash
cd janus-rl/src/open_r1

export ACCELERATE_CONFIG=../../recipes/accelerate_configs/zero2.yaml
export GRPO_CONFIG=../../recipes/t2i_generation/grpo.yml
export NUM_PROCESSES=8

accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  --num_processes $NUM_PROCESSES \
  grpo_t2i.py \
  --config $GRPO_CONFIG
```

Additionally, you can use the following command for RL on **Image Editing**:

```bash
cd janus-rl/src/open_r1

export ACCELERATE_CONFIG=../../recipes/accelerate_configs/zero2.yaml
export GRPO_CONFIG=../../recipes/image_editing/grpo.yml
export NUM_PROCESSES=8

accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  --num_processes $NUM_PROCESSES \
  grpo_editing.py \
  --config $GRPO_CONFIG
```