import os
import sys
import subprocess

def install_dependencies():
    """Install required packages like tiktoken if not available."""
    try:
        import tiktoken
    except ImportError:
        print("tiktoken not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])

def get_platform():
    """Detect if the script is running on Google Colab, Kaggle, or Local."""
    if 'google.colab' in sys.modules:
        return "colab"
    elif os.path.exists('/kaggle'):
        return "kaggle"
    else:
        return "local"

def get_file_path(platform, input_file):
    """Return the appropriate file path based on the platform."""
    if platform == "colab":
        # Assuming vulkan dataset is stored in Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
        return f"/content/drive/MyDrive/{input_file}"
    elif platform == "kaggle":
        # Assuming vulkan dataset is uploaded as a Kaggle dataset
        return f"/kaggle/input/{input_file}"
    else:
        # Local environment
        return f"./{input_file}"

def main():
    # Detect platform
    platform = get_platform()
    print(f"Running on {platform}...")

    # Install dependencies
    install_dependencies()

    # Define the input file and dataset name
    dataset_name = "vulkan"
    input_file = "vulkan_dataset/vulkan_spec.txt"

    # Get the correct file path
    input_file_path = get_file_path(platform, input_file)

    # Run the shards creation script
    shard_command = f"python shards_textinput.py --dataset {dataset_name} --input_file {input_file_path}"
    os.system(shard_command)

    # Run the training script
    train_command = f"python train_gpt2.py --dataset {dataset_name}"
    os.system(train_command)

if __name__ == "__main__":
    main()
