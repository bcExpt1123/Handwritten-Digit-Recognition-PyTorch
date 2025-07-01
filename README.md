# Simple Neural Network for MNIST Handwritten Digit Recognition

This notebook demonstrates a simple neural network (NN) implemented using PyTorch for classifying handwritten digits from the MNIST dataset. The model is trained and evaluated on the MNIST dataset, achieving relatively high accuracy.

## Overview

The [notebook](https://www.kaggle.com/code/bcexpt1123/handwritten-1-0-0) covers the following steps:

1.  **Data Loading and Preprocessing**: Loading the MNIST dataset from a directory, applying transformations, and creating data loaders.
2.  **Model Definition**: Defining a simple neural network with fully connected layers.
3.  **Training**: Training the model using the training dataset and an optimization algorithm.
4.  **Evaluation**: Evaluating the trained model on the test dataset.
5.  **Visualization**: Displaying sample predictions from the test dataset.
6.  **Saving and Loading the Model**: Saving the trained model for later use and loading it back.

## Dependencies

*   torch
*   torchvision
*   matplotlib
*   PIL (Pillow)
*   os

You can install these packages using pip:

```bash
pip install torch torchvision matplotlib Pillow
```

## Dataset

The [MNIST dataset](https://www.kaggle.com/datasets/alexanderyyy/mnist-png) is expected to be structured as follows:

```
mnist_png/
├── train/
│   ├── 0/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   ├── 1/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   └── ...
└── test/
    ├── 0/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    ├── 1/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── ...
```

Each digit (0-9) has its own directory under both `train` and `test` directories.

The dataset is loaded from the `/kaggle/input/mnist-png/mnist_png` directory, which is a common location for Kaggle kernels.

## Model Architecture

The neural network architecture is defined in the `SimpleNN` class:

*   **Input Layer**: Flattens the 28x28 images into a 784-dimensional vector.
*   **Fully Connected Layers**:
    *   `fc1`: 784 input features, 128 output features.
    *   `fc2`: 128 input features, 64 output features.
    *   `fc3`: 64 input features, 10 output features (for the 10 digits).
*   **ReLU Activation**: Applied after `fc1` and `fc2` to introduce non-linearity.

## Usage

1.  **Import Libraries**:

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    ```
2.  **Define the Neural Network**:

    ```python
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, 28*28)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    ```
3.  **Create a Custom Dataset**:

    ```python
    from torch.utils.data import Dataset, DataLoader
    import os

    class SimpleDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.images = []
            self.labels = []

            for label in os.listdir(root_dir):
                label_dir = os.path.join(root_dir, label)
                print(label_dir)
                if os.path.isdir(label_dir):
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        self.images.append(img_path)
                        self.labels.append(int(label))


        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert('L')
            if self.transform:
                img = self.transform(img)
            label = self.labels[idx]
            return img, label
    ```
4.  **Define Data Transformations**:

    ```python
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ```
5.  **Load the Dataset**:

    ```python
    root_dir = '/kaggle/input/mnist-png/mnist_png'
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")

    train_dataset = SimpleDataset(root_dir=train_dir, transform=transform)
    test_dataset = SimpleDataset(root_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    ```
6.  **Train the Model**:

    ```python
    def train_model(model, train_loader, criterion, optimizer, epoches=5):
        for epoch in range(epoches):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{epoches}], Loss: {loss.item():.4f}")
    ```
7.  **Evaluate the Model**:

    ```python
    def evaluate_model(model, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total:.2f}%")
    ```
8.  **Visualize Predictions**:

    ```python
    def visualize_predictions(model, test_loader):
        dataiter = iter(test_loader)
        images, labels = next(dataiter)

        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

        # Plotting
        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(f'Pred: {preds[i].item()}, True: {labels[i].item()}')
            plt.axis('off')
        plt.show()
    ```
9.  **Instantiate, Train, and Evaluate**:

    ```python
    model = SimpleNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epoches=5)

    evaluate_model(model, test_loader)
    ```
10. **Visualize Predictions**:

    ```python
    visualize_predictions(model, test_loader)
    ```

## Training

The model is trained for 5 epochs using the Adam optimizer and CrossEntropyLoss. The training progress is printed for each epoch, showing the loss value.

```
Epoch [1/5], Loss: 0.1367
Epoch [2/5], Loss: 0.0314
Epoch [3/5], Loss: 0.2148
Epoch [4/5], Loss: 0.0144
Epoch [5/5], Loss: 0.0072
Accuracy: 96.90%
```

## Evaluation

The trained model achieves an accuracy of approximately 96.90% on the test dataset.

## Saving and Loading the Model

*   **Save the Model**:

    ```python
    torch.save(model, 'simple_nn.pth')
    ```
*   **Load the Model**:

    ```python
    model.load_state_dict(torch.load('simple_nn.pth'))
    model.eval()
    ```

## Visualization

The notebook includes a function to visualize the predictions on a small sample of the test dataset. It displays 10 images along with their predicted and true labels.

## Additional Notes

*   The notebook uses a simple neural network architecture for demonstration purposes. More complex architectures (e.g., CNNs) can achieve higher accuracy.
*   The number of epochs and batch size can be adjusted to optimize training performance.
*   The dataset is loaded from a specific directory (`/kaggle/input/mnist-png/mnist_png`), which may need to be modified based on your environment.
