pip install tiktoken
python shards_textinput.py --dataset vulkan --input_file ./vulkan_dataset/vulkan_spec.txt
python train_gpt2.py --dataset vulkan

