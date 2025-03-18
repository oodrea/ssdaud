import torchvision.models as models
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
)

# Function to load the selected model
def load_model(choice):
    if choice == "efficientnet":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif choice == "mobilenet":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif choice == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    else:
        print("Invalid choice. Please select from: efficientnet, mobilenet, shufflenet.")
        return None
    return model

# User selects the model
print("Select a model to print:\n1. EfficientNet\n2. MobileNetV3\n3. ShuffleNetV2")
option = input("Enter the number of your choice: ").strip()

# Map user input to model names
model_dict = {"1": "efficientnet", "2": "mobilenet", "3": "shufflenet"}
selected_model = model_dict.get(option)

# Load the chosen model
model = load_model(selected_model)

if model:
    # Define the file name based on the model choice
    output_file = f"{selected_model}_model.txt"

    # Save model architecture and number of parameters to a text file
    with open(output_file, "w") as f:
        # Write model architecture
        f.write(str(model) + "\n\n")

        # Compute and write the number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"Total Parameters: {total_params}\n")

    print(f"Model architecture and parameter count saved to {output_file}")
