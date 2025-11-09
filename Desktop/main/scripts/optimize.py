import os
import torch
import torch.optim as optim
import yaml
import argparse
import logging
from datetime import datetime
import optuna
import numpy as np

from ..models.custom_model import CustomDetectionModel, YOLOLoss
from ..utils.data_loader import create_data_loaders


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def objective(trial, config, data_loaders, device, logger):
    """
    Objective function for Bayesian optimization.

    Args:
        trial: Optuna trial object
        config: Configuration dictionary
        data_loaders: Data loaders
        device: Device to run on
        logger: Logger instance

    Returns:
        float: Validation loss to minimize
    """
    # Sample hyperparameters
    lr = trial.suggest_float('lr', config['lr_range'][0], config['lr_range'][1], log=True)
    batch_size = trial.suggest_int('batch_size', config['batch_size_range'][0], config['batch_size_range'][1])
    pruning_ratio = trial.suggest_float('pruning_ratio', config['pruning_ratio_range'][0], config['pruning_ratio_range'][1])

    logger.info(f"Trial {trial.number}: lr={lr:.6f}, batch_size={batch_size}, pruning_ratio={pruning_ratio:.2f}")

    # Create model
    num_classes = data_loaders['datasets']['train'].get_num_classes()
    model = CustomDetectionModel(num_classes=num_classes)
    model.to(device)

    # Create loss function and optimizer
    criterion = YOLOLoss(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create data loaders with sampled batch size
    train_loader = torch.utils.data.DataLoader(
        data_loaders['datasets']['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        data_loaders['datasets']['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Training loop (simplified for optimization)
    num_epochs = 10  # Reduced epochs for optimization
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch['images'].to(device)
            targets = batch['targets']

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                targets = batch['targets']

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Apply pruning if ratio > 0
        if pruning_ratio > 0:
            example_input = torch.randn(1, 3, config['img_size'], config['img_size']).to(device)
            pruning_stats = model.prune_model(pruning_ratio=pruning_ratio, example_input=example_input)

            # Fine-tune after pruning (1 epoch)
            model.train()
            for batch in train_loader:
                images = batch['images'].to(device)
                targets = batch['targets']

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                break  # Just one batch for fine-tuning

        # Update best loss
        if val_loss < best_loss:
            best_loss = val_loss

        # Report intermediate result
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    logger.info(f"Trial {trial.number} completed: Best validation loss = {best_loss:.4f}")
    return best_loss


def run_bayesian_optimization(config, data_loaders, device, logger):
    """
    Run Bayesian optimization to find best hyperparameters.

    Args:
        config: Configuration dictionary
        data_loaders: Data loaders
        device: Device to run on
        logger: Logger instance

    Returns:
        dict: Best hyperparameters and results
    """
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, config, data_loaders, device, logger),
        n_trials=config['bayesian_opt_trials']
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Optimization completed!")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best validation loss: {best_value:.4f}")

    return {
        'best_params': best_params,
        'best_value': best_value,
        'study': study
    }


def save_optimization_results(results, output_dir, logger):
    """
    Save optimization results.

    Args:
        results: Optimization results
        output_dir: Output directory
        logger: Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save best parameters
    params_path = os.path.join(output_dir, 'best_params.yaml')
    with open(params_path, 'w') as f:
        yaml.dump(results['best_params'], f)

    # Save study results
    study_path = os.path.join(output_dir, 'optimization_study.pkl')
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(results['study'], f)

    logger.info(f"Optimization results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for model hyperparameters')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='optimization_results', help='Output directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config['log_dir'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loaders
    data_loaders = create_data_loaders(
        args.data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size']
    )

    # Run optimization
    results = run_bayesian_optimization(config, data_loaders, device, logger)

    # Save results
    save_optimization_results(results, args.output_dir, logger)

    logger.info("Bayesian optimization completed!")


if __name__ == "__main__":
    main()
