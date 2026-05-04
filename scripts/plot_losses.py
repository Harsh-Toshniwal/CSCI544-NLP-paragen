"""
Plot training losses and metrics from CSV file.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_losses(
    losses_path: str,
    output_dir: str = None,
    figsize: tuple = (14, 5),
):
    """
    Plot training and validation losses.

    Args:
        losses_path: Path to losses CSV file
        output_dir: Directory to save plots (default: same as losses_path)
        figsize: Figure size (width, height)
    """
    losses_path = Path(losses_path)
    
    if not losses_path.exists():
        logger.error(f"Losses file not found: {losses_path}")
        return

    # Load losses
    df = pd.read_csv(losses_path)
    logger.info(f"Loaded losses from {losses_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nColumns: {df.columns.tolist()}")
    logger.info(f"\nFirst few rows:\n{df.head()}")

    # Set output directory
    if output_dir is None:
        output_dir = losses_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate train and val data
    train_df = df[df["stage"] == "train"].reset_index(drop=True)
    val_df = df[df["stage"] == "val"].reset_index(drop=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Loss
    ax1 = axes[0]
    if len(train_df) > 0:
        ax1.plot(
            train_df["epoch"],
            train_df["loss"],
            marker="o",
            label="Train Loss",
            linewidth=2,
        )
    if len(val_df) > 0:
        ax1.plot(
            val_df["epoch"],
            val_df["loss"],
            marker="s",
            label="Val Loss",
            linewidth=2,
        )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2 = axes[1]
    if len(train_df) > 0 and train_df["acc"].notna().any():
        ax2.plot(
            train_df["epoch"],
            train_df["acc"],
            marker="o",
            label="Train Accuracy",
            linewidth=2,
        )
    if len(val_df) > 0 and val_df["acc"].notna().any():
        ax2.plot(
            val_df["epoch"],
            val_df["acc"],
            marker="s",
            label="Val Accuracy",
            linewidth=2,
        )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    if len(val_df) > 0 and val_df["acc"].notna().any():
        ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "losses_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")

    # Create separate detailed plots
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    if len(train_df) > 0:
        ax3.plot(
            train_df["epoch"],
            train_df["loss"],
            marker="o",
            label="Train Loss",
            linewidth=2.5,
            markersize=8,
        )
    if len(val_df) > 0:
        ax3.plot(
            val_df["epoch"],
            val_df["loss"],
            marker="s",
            label="Val Loss",
            linewidth=2.5,
            markersize=8,
        )
    
    # Add best val loss marker
    if len(val_df) > 0 and val_df["loss"].notna().any():
        best_idx = val_df["loss"].idxmin()
        best_epoch = val_df.loc[best_idx, "epoch"]
        best_loss = val_df.loc[best_idx, "loss"]
        ax3.scatter([best_epoch], [best_loss], color="red", s=200, marker="*", 
                   label=f"Best Val Loss ({best_loss:.4f})", zorder=5)
    
    ax3.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax3.set_title("Detailed Training and Validation Loss", fontsize=15, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    detailed_plot_path = output_dir / "losses_detailed.png"
    plt.savefig(detailed_plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Detailed plot saved to {detailed_plot_path}")

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("TRAINING STATISTICS")
    logger.info("="*60)
    
    if len(train_df) > 0:
        logger.info(f"\nTrain Loss - Min: {train_df['loss'].min():.4f}, "
                   f"Max: {train_df['loss'].max():.4f}, "
                   f"Mean: {train_df['loss'].mean():.4f}")
        if train_df["acc"].notna().any():
            logger.info(f"Train Acc - Min: {train_df['acc'].min():.4f}, "
                       f"Max: {train_df['acc'].max():.4f}, "
                       f"Mean: {train_df['acc'].mean():.4f}")
    
    if len(val_df) > 0:
        logger.info(f"\nVal Loss - Min: {val_df['loss'].min():.4f}, "
                   f"Max: {val_df['loss'].max():.4f}, "
                   f"Mean: {val_df['loss'].mean():.4f}")
        if val_df["acc"].notna().any():
            logger.info(f"Val Acc - Min: {val_df['acc'].min():.4f}, "
                       f"Max: {val_df['acc'].max():.4f}, "
                       f"Mean: {val_df['acc'].mean():.4f}")
    
    logger.info("="*60 + "\n")

    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Plot training losses")
    parser.add_argument(
        "--losses_path",
        type=str,
        default="checkpoints/mistral_encoder_lora/losses.csv",
        help="Path to losses CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[14, 5],
        help="Figure size (width height)",
    )

    args = parser.parse_args()

    plot_losses(
        losses_path=args.losses_path,
        output_dir=args.output_dir,
        figsize=tuple(args.figsize),
    )


if __name__ == "__main__":
    main()
