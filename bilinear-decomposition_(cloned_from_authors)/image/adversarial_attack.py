import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    device = batch.device
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def pgd_attack(model, data, target, criterion, epsilon=0.3, num_iter=40, clamp_min=0.0, clamp_max=1.0):
    # As per Madry et al. (2018)
    alpha = (2.5 * epsilon) / num_iter

    original_data = data.clone().detach()
    perturbed_data = data.clone().detach()

    # Iteratively perturb the data in the direction of the gradient
    for i in range(num_iter):
        perturbed_data.requires_grad = True

        outputs = model(perturbed_data)
        loss = criterion(outputs, target)
        loss.backward()

        perturbed_data_grad = perturbed_data.grad
        sign_input_grad = torch.sign(perturbed_data_grad)
        perturbation = alpha * sign_input_grad
        
        perturbed_data = perturbed_data.detach() + perturbation

        # Project back to epsilon-ball
        delta = torch.clamp(
            perturbed_data - original_data,
            min=-epsilon,
            max=epsilon
        )

        perturbed_data = original_data + delta

        perturbed_data = torch.clamp(perturbed_data, clamp_min, clamp_max)

    return perturbed_data

def fgsm_attack(image, data_grad, epsilon = 0.25):
    # Get the sign of the data gradient (element-wise)
    sign_data_grad = torch.sign(data_grad)
    eta = epsilon * sign_data_grad
    # Create the perturbed image, scaled by epsilon
    perturbed_image = image + eta
    # Make sure values stay within valid range
    perturbed_image = torch.clamp(perturbed_image, min=image.min(), max=image.max())
    return perturbed_image
    
def fgsm_loss(model, criterion, inputs, labels, alpha=0.5, epsilon=0.25, return_preds = True):
    inputs.requires_grad = True
    
    original_outputs = model(inputs)
    # Calculate the loss for the original image
    original_loss = criterion(original_outputs, labels)
   
    input_grad = torch.autograd.grad(original_loss, inputs, retain_graph=True, create_graph=True)[0]
    # Calculate the perturbation
    perturbation = epsilon * torch.sign(input_grad)
    perturbed_image = inputs + perturbation

    perturbed_outputs = model(perturbed_image)
    # Calculate the loss for the perturbed image
    perturbation_loss = criterion(perturbed_outputs, labels)

    # Combine the two losses
    loss = (alpha * original_loss) + ((1 - alpha) * perturbation_loss)

    if return_preds:
        return loss, original_outputs
    else:
        return loss

def test_attack(model, test_dataset, attack_function, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    correct = 0
    criterion = nn.CrossEntropyLoss()
    adv_examples = []

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # important for attack
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1] 

            # If the initial prediction is wrong, don't attack
            if init_pred.item() != target.item():
                continue

            loss = F.nll_loss(output, target)
            model.zero_grad()
            
            if attack_function == "FGSM": 
                # Get the correct gradients wrt the data
                loss.backward()
                data_grad = data.grad
                # Perturb the data using the FGSM attack
                perturbed_data = fgsm_attack(data, data_grad)
                # Re-classify the perturbed image
                output = model(perturbed_data)

            elif attack_function == "PGD":
                perturbed_data = pgd_attack(
                    model=model,
                    data=data,
                    target=target,
                    criterion=criterion
                )

                output = model(perturbed_data)

            else:
                print(f"Unknown attack: {attack_function}")

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] 
            if final_pred.item() == target.item():
                correct += 1
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    original_data = data.squeeze().detach().cpu()
                    adv_ex = perturbed_data.squeeze().detach().cpu()
                    adv_examples.append( (init_pred.item(), 
                                        final_pred.item(),
                                        original_data, 
                                        adv_ex) )

    # Calculate final accuracy
    final_acc = correct/float(len(test_loader))
    print(f"Attack {attack_function}, \nTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples
