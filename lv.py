import torch

def calculate_lv(tensor):

    # Iterate over each layer (dimension 0)
    total_l2_sum = 0.0
    for layer in tensor:
        # Calculate L2 norm for each 224x224 matrix
        for row in range(224):
            for col in range(224):
                # Get the current element
                current_element = tensor[:, row, col]

                # Get the right and bottom neighbors
                right_neighbor = tensor[:, row, col + 1] if col < 223 else 0.0
                bottom_neighbor = tensor[:, row + 1, col] if row < 223 else 0.0

                # Calculate L2 norm
                l2_norm = torch.norm(current_element - right_neighbor) + torch.norm(current_element - bottom_neighbor)

                # Add to the total sum
                total_l2_sum += l2_norm.item()

    return total_l2_sum





