"""
AWS MLA-C01: PyTorch Training Script for SageMaker Script Mode

This script demonstrates PyTorch training using SageMaker Script Mode:
  1. Parse hyperparameters (epochs, learning rate, batch size, hidden dimensions)
  2. Load training data from SageMaker's default data directories
  3. Define neural network architecture (ChurnNet with dropout & batch normalization)
  4. Implement training loop with loss tracking and validation
  5. Save trained model for deployment
  6. Implement model_fn, input_fn, predict_fn for SageMaker hosting

Key MLA-C01 Concepts:
  - Script Mode: Run custom training code without modifying for SageMaker
  - Hyperparameter Handling: Parse from command-line arguments
  - Data Paths: SageMaker mounts data at /opt/ml/input/data/
  - Model Artifacts: Save to /opt/ml/model/
  - Hosted Inference: Implement handler functions for predictions
  - Distributed Training: Integrated with SageMaker's distributed training
"""

import argparse
import logging
import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: Define Custom Dataset Class
# ============================================================================

class ChurnDataset(Dataset):
    """
    Custom PyTorch Dataset for churn prediction.
    Loads CSV data and converts to PyTorch tensors.
    """

    def __init__(self, features, labels):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx):
        """Return feature and label for index."""
        return self.features[idx], self.labels[idx]


# ============================================================================
# STEP 2: Define Neural Network Architecture
# ============================================================================

class ChurnNet(nn.Module):
    """
    Neural network for customer churn prediction.

    Architecture:
      Input → Dense(hidden_dim) → ReLU → Dropout
           → Dense(hidden_dim) → ReLU → Dropout
           → Dense(1) → Sigmoid
           → Output (probability)

    Key components:
      - Input features: TBD (based on data)
      - Hidden layers: 2 layers with hidden_dim units each
      - Dropout: Regularization to prevent overfitting
      - Output: 1 unit with sigmoid (binary classification)
    """

    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        """
        Initialize network layers.

        Args:
            input_dim: Number of input features
            hidden_dim: Number of units in hidden layers
            dropout_rate: Dropout probability (0.0-1.0)
        """
        super(ChurnNet, self).__init__()

        # First hidden layer
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Second hidden layer
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Output layer
        self.dense3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through network.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            y: Output tensor (batch_size, 1) - probabilities in [0, 1]
        """
        # First hidden layer with ReLU and dropout
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second hidden layer with ReLU and dropout
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Output layer with sigmoid for probability
        x = self.dense3(x)
        x = self.sigmoid(x)

        return x


# ============================================================================
# STEP 3: Training and Validation Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer (Adam, SGD, etc.)
        device: torch.device (cpu or cuda)

    Returns:
        Average loss for the epoch
    """
    model.train()  # Set to training mode (enables dropout)
    total_loss = 0.0
    batch_count = 0

    for batch_features, batch_labels in train_loader:
        # Move data to device (GPU if available)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device).unsqueeze(1)  # Shape: (batch_size, 1)

        # Forward pass
        predictions = model(batch_features)

        # Compute loss
        loss = criterion(predictions, batch_labels)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        # Track loss
        total_loss += loss.item()
        batch_count += 1

    average_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return average_loss


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set.

    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: torch.device

    Returns:
        Average loss and accuracy
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)

            # Forward pass
            predictions = model(batch_features)

            # Compute loss
            loss = criterion(predictions, batch_labels)
            total_loss += loss.item()

            # Compute accuracy (threshold at 0.5)
            predicted_classes = (predictions > 0.5).float()
            correct += (predicted_classes == batch_labels).sum().item()
            total += batch_labels.size(0)

    average_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return average_loss, accuracy


# ============================================================================
# STEP 4: Model Handler Functions for SageMaker Hosting
# ============================================================================

def model_fn(model_dir):
    """
    Load model for inference.
    Called once when hosting container starts.

    Args:
        model_dir: Path to model artifacts (default: /opt/ml/model)

    Returns:
        Loaded model ready for prediction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChurnNet(input_dim=6, hidden_dim=64, dropout_rate=0.3)

    # Load model weights from pickle
    model_path = os.path.join(model_dir, "churn_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def input_fn(request_body, content_type="application/json"):
    """
    Parse input data for inference.
    Converts JSON request to model input.

    Args:
        request_body: Request body (JSON string)
        content_type: Content type (application/json)

    Returns:
        Tensor ready for model inference
    """
    if content_type == "application/json":
        input_data = json.loads(request_body)
        features = torch.FloatTensor([input_data["features"]])
        return features
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Generate prediction using model.

    Args:
        input_data: Model input (tensor)
        model: Loaded model

    Returns:
        Prediction (probability)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)

    with torch.no_grad():
        prediction = model(input_data)

    return prediction.cpu().numpy()


def output_fn(prediction, content_type="application/json"):
    """
    Format model output for response.

    Args:
        prediction: Model prediction (numpy array)
        content_type: Response content type

    Returns:
        Formatted output (JSON string)
    """
    if content_type == "application/json":
        output = {
            "churn_probability": float(prediction[0][0]),
            "predicted_churn": "Yes" if prediction[0][0] > 0.5 else "No"
        }
        return json.dumps(output)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# ============================================================================
# STEP 5: Main Training Function
# ============================================================================

def main():
    """Main training entry point."""

    logger.info("PyTorch Training Script Started")

    # Parse hyperparameters from command-line arguments
    # SageMaker passes hyperparameters as command-line arguments
    parser = argparse.ArgumentParser()

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.3)

    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--training", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--validation", type=str, default="/opt/ml/input/data/validation")

    args, _ = parser.parse_known_args()

    logger.info(f"Hyperparameters:")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  learning_rate: {args.learning_rate}")
    logger.info(f"  hidden_dim: {args.hidden_dim}")
    logger.info(f"  dropout_rate: {args.dropout_rate}")

    # Determine device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load and Prepare Data
    # ========================================================================

    logger.info("\nLoading training data...")

    # In SageMaker Script Mode, data is mounted at:
    # /opt/ml/input/data/training/
    # /opt/ml/input/data/validation/

    # Load training data
    train_csv = os.path.join(args.training, "train.csv")
    train_data = pd.read_csv(train_csv, header=None)

    # Assume: first column is label, rest are features
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values

    logger.info(f"Training data shape: {X_train.shape}")

    # Load validation data if available
    val_csv = os.path.join(args.validation, "validation.csv")
    if os.path.exists(val_csv):
        val_data = pd.read_csv(val_csv, header=None)
        X_val = val_data.iloc[:, 1:].values
        y_val = val_data.iloc[:, 0].values
        logger.info(f"Validation data shape: {X_val.shape}")
    else:
        # Use portion of training data if no validation set
        split_idx = int(0.2 * len(X_train))
        X_val = X_train[:split_idx]
        y_val = y_train[:split_idx]
        X_train = X_train[split_idx:]
        y_train = y_train[split_idx:]
        logger.info(f"Using 20% of training data for validation")
        logger.info(f"Updated train shape: {X_train.shape}, val shape: {X_val.shape}")

    # Feature normalization using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    logger.info("Feature scaling applied")

    # Create PyTorch datasets and data loaders
    train_dataset = ChurnDataset(X_train, y_train)
    val_dataset = ChurnDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set > 0 for multi-worker loading on multi-core systems
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"DataLoaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # ========================================================================
    # Initialize Model, Loss, and Optimizer
    # ========================================================================

    input_dim = X_train.shape[1]
    logger.info(f"\nInitializing model with input_dim={input_dim}")

    model = ChurnNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
    )
    model.to(device)

    # Binary cross-entropy loss (standard for binary classification)
    criterion = nn.BCELoss()

    # Adam optimizer (adaptive learning rate, generally good default)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Loss: {criterion.__class__.__name__}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")

    # ========================================================================
    # Training Loop
    # ========================================================================

    logger.info(f"\nStarting training for {args.epochs} epochs...")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Log metrics
        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    logger.info("Training completed")

    # ========================================================================
    # Save Model
    # ========================================================================

    logger.info(f"\nSaving model to {args.model_dir}")

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(args.model_dir, "churn_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model weights saved to {model_path}")

    # Save scaler for inference
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save training metadata
    metadata = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout_rate": args.dropout_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "final_val_accuracy": float(val_accuracies[-1]),
        "training_timestamp": datetime.now().isoformat(),
    }

    metadata_path = os.path.join(args.model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    # ========================================================================
    # Final Summary
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Final Training Loss: {train_losses[-1]:.4f}")
    logger.info(f"Final Validation Loss: {val_losses[-1]:.4f}")
    logger.info(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")
    logger.info(f"Best Validation Accuracy: {max(val_accuracies):.4f}")
    logger.info("=" * 70)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
