import re
import matplotlib.pyplot as plt

# Function to parse the log file
def parse_log_file(log_file_path):
    training_steps = []
    training_losses = []
    validation_steps = []
    validation_losses = []

    with open(log_file_path, 'r') as file:
        for line in file:
            if "train" in line:
                match = re.search(r'(\d+) train ([\d.]+)', line)
                if match:
                    training_steps.append(int(match.group(1)))
                    training_losses.append(float(match.group(2)))
            elif "val" in line:
                match = re.search(r'(\d+) val ([\d.]+)', line)
                if match:
                    validation_steps.append(int(match.group(1)))
                    validation_losses.append(float(match.group(2)))

    return training_steps, training_losses, validation_steps, validation_losses

# Main script to visualize loss
def main():
    # Path to your log file
    log_file_path = 'log/log.txt'

    # Parse the log file
    train_steps, train_losses, val_steps, val_losses = parse_log_file(log_file_path)

    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_losses, label='Training Loss', color='b')
    plt.plot(val_steps, val_losses, label='Validation Loss', color='r')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    # Show the plot in blocking mode
    plt.show(block=True)

if __name__ == "__main__":
    main()
