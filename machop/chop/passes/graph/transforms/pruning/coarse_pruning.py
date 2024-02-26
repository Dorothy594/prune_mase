import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def channel_pruning(model, sparsity):
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            num_channels = layer.weight.size(0)
            num_channels_to_keep = max(1, int(num_channels * (1 - sparsity)))

            # Update the fully connected layer based on the number of channels to keep
            model.fc = nn.Linear(num_channels_to_keep * 8 * 8, 10)

            _, indices = torch.topk(torch.sum(torch.abs(layer.weight.data), dim=(1, 2, 3)), num_channels_to_keep)
            mask = torch.zeros_like(layer.weight.data)
            mask[indices, :, :, :] = 1
            layer.weight.data *= mask


def kernel_pruning(model, layer_name, sparsity):
    for name, layer in model.named_children():
        if name == layer_name and isinstance(layer, nn.Conv2d):
            num_filters = layer.weight.size(0)
            num_filters_to_keep = max(1, int(num_filters * (1 - sparsity)))

            _, indices = torch.topk(torch.sum(torch.abs(layer.weight.data), dim=(1, 2, 3)), num_filters_to_keep)
            mask = torch.zeros_like(layer.weight.data[:, 0, 0, 0])
            mask[indices] = 1
            layer.weight.data *= mask.view(-1, 1, 1, 1)
            layer.bias.data *= mask


# Create an instance of the model
model = SimpleCNN()

# Run the forward pass with the input tensor
x = torch.randn(1, 3, 32, 32)

# Print the original model structure and parameters
print("Original Model:")
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

# Channel pruning with 80% sparsity
channel_pruning(model, sparsity=0.8)

# Print the pruned model structure and parameters
print("\nChannel Pruned Model:")
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

# 卷积核剪枝，保留80%的卷积核
kernel_pruning(model, 'conv1', sparsity=0.8)

# 打印剪枝后的模型结构和参数数量
print("\nKernel Pruned Model:")
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))