from typing import Any
import qai_hub
import torch
import pandas as pd
import numpy as np
from utils.mobileone import reparameterize_model


def read_ground_truth_from_csv(csv_file):
    # Read the CSV file into a DataFrame
    ground_truth_data = pd.read_csv(csv_file)

    # Drop rows where 'class_index' is NaN (empty)
    ground_truth_data = ground_truth_data.dropna(subset=['class_index'])

    # Return the list of class indices
    return ground_truth_data['class_index'].tolist()


def evaluate_track1(output_array, output_dir, ground_truth_dir):
    # Read the ground truth indices from the provided CSV file
    ground_truth_indices = read_ground_truth_from_csv(ground_truth_dir)

    # Initialize counters to track correct predictions
    correct = 0
    total = len(ground_truth_indices)

    # Compare predictions with ground truth
    for i, result in enumerate(output_array):
        # Apply softmax to the model output to get probabilities
        softmax_results = softmax(result)

        # Get the top prediction (class with highest probability)
        top_prediction = np.argmax(softmax_results)

        # Compare the top prediction to the ground truth
        if top_prediction == ground_truth_indices[i]:
            correct += 1

    accuracy = correct / total
    print(f"Correct predictions: {correct}/{total}")
    return accuracy
def run_inference(model, device, input_dataset):
    """Submits an inference job for the model and returns the output data."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1"
    )
    return inference_job.download_output_data()
def compile_model(model, device, input_shape):
    """Submits a compile job for the model and returns the job instance."""
    return qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs={"image": input_shape},
        options="--target_runtime tflite"
    )

# model = models.ViT_B_16_Weights
class Preprocessed_model(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()

        self.model = torch.load(pretrained_model, weights_only=False, map_location=torch.device('cpu'))
        self.preprocess = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        return x

# Define target device
device = qai_hub.Device("Snapdragon 8 Elite QRD")

model_path = "../checkpoint/fastvit_mci0_ra_ls_sam_v6_lr_0.0001_bs_128_best.pth"

# Define input shape
input_shape = (1, 3, 224, 224)
model = Preprocessed_model(model_path)
model.eval()
re_model = reparameterize_model(model)

pt_model = torch.jit.trace(re_model, torch.randn(input_shape))
 # Replace with the actual model instance
"""
Example:
# Construct your own model
class PreprocessedMobileNetV2(torch.nn.Module):
    ...

model = PreprocessedMobileNetV2()

or

model = qai_hub.get_model("model_id")
"""

# Compile the model
compile_job = compile_model(pt_model, device, input_shape)

# Sharing the model
compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
compile_job.set_name(f"f_v6_ep89_LPCVC25")

# Output compiled job ID
print(f"Model compiled successfully with ID {compile_job.job_id}")


# Replace with actual compiled job ID and dataset ID
compiled_id = compile_job.job_id # Set the compiled job ID
# shenbo d67j3wlz9  konglh dn7x4zo52
input_dataset = qai_hub.get_dataset("d67j3wlz9")  # Set the dataset ID (refer to upload_dataset.py)

# Retrieve the compiled model
job = qai_hub.get_job(compiled_id)
compiled_model = job.get_target_model()

# Run inference
print(f"Running inference for model {compiled_model.model_id} on device {device.name}")
inference_output = run_inference(compiled_model, device, input_dataset)


# Extract output
output_array = inference_output.get('output_0')

# Evaluate the model
print(evaluate_track1(output_array, "output", "key.csv"))
