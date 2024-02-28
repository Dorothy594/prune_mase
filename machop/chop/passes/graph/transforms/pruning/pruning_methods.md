# Pruning Methods
## L1 and L2
## Gradient-based Pruning
### Implementation
1. Train the model and compute gradients for each weight.
2. Sort the gradients or set a gradient threshold.
3. Based on the sorting or threshold, selectively prune weights with smaller gradients.
### Principle
Pruning based on gradients assumes that weights with smaller gradients during training may contribute less to themodel. Therefore, pruning is applied to reduce the model size. This is becauuse, during training, weights with smaller gradients may have a smaller impact on optimizing the loss function.
### Coding
```
import torch
import torch.nn as nn
import torch.optim as optim

def train_and_prune(model, input_data, target, gradient_threshold=0.001, learning_rate=0.01):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Forward propagation, compute loss, back propagation, and gradient update
    output = model(input_data)
    loss = criterion(output, target)

    optimizer.zero_grad()  # Clear gradient
    loss.backward()       # Back propagation, compute gradient
    optimizer.step()       # Update weights

    # Pruning before gradient update
    mask = []
    for param in model.parameters():
        param_mask = param.grad.abs() > gradient_threshold
        param.grad.data[param_mask] = 0
        mask.append(param_mask)

    return mask

# Usage example:
input_data = torch.randn((1, 10))
target = torch.randn((1, 1))

# Create an instance of the model
model = SimpleModel()

# Train and prune the model, and get the mask
pruning_mask = train_and_prune(model, input_data, target)
```

## K-means Clustering
### Implementation
1. Cluster the weights using K-means clustering, dividing weights into K clusters (cluster centers).
2. Calculate the center value for each cluster.
3. Selectively prune based on the center values. You can set a threshold for cluster centers and prune clusters with smaller center values.
### Principle
The K-means clustering pruning method assumes that by clustering weights, smaller clusters can be identified, and weights within these clusters can be pruned. This method aims to represent the "importance" of clusters based on the size of their cluster center values.

**Cluster Feature Extraction**: K-means clustering is used to group weights into clusters with similar characteristics. The formation of these clusters is based on the similarity between weights.

**Importance Representation**: The decision for pruning is based on the center value of each cluster. Clusters with smaller center values may contain weights that have a smaller impact on the model, and therefore, can be selectively pruned. This is based on the assumption that smaller center values indicate weights with less contribution to the model.

**Threshold Strategy**: The decision for pruning can be accomplished by setting a threshold on the center values. If the center value of a cluster is below the threshold, the cluster can be selectively pruned.

### Coding
```
from sklearn.cluster import KMeans

def kmeans(tensor: torch.Tensor, info: dict, num_clusters: int, sparsity: float) -> torch.Tensor:
    # Flatten tensor values for clustering
    flattened_values = tensor.flatten().view(-1, 1).cpu().numpy()

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flattened_values)

    # Get cluster centers and prune values in clusters with smaller magnitudes
    cluster_centers = torch.tensor(kmeans.cluster_centers_).to(tensor.device)
    threshold = torch.quantile(cluster_centers.abs().flatten(), sparsity)
    mask = (cluster_centers.abs() > threshold).to(torch.bool).to(tensor.device)

    # Apply mask to original tensor
    mask_tensor = torch.zeros_like(tensor, dtype=torch.bool).to(tensor.device)
    for i, cluster_id in enumerate(kmeans.labels_):
        mask_tensor.view(-1)[i] = mask[cluster_id]

    return mask_tensor
```

## Structured Pruning (Channel Pruning & Kernel Pruning)