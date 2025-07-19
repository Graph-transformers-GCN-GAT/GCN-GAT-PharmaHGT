import matplotlib.pyplot as plt
import os

# def plot_loss_and_scatter(train_losses, val_losses, train_mses, val_mses, train_r2s, val_r2s, fold, model_name, path):
#     epochs = range(1, len(train_losses) + 1)
#     plt.figure(figsize=(15, 5))

#     # Plotting the losses
#     plt.subplot(1, 3, 1)
#     plt.plot(epochs, train_losses, 'bo-', label='Training loss')
#     plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title(f'{model_name} - Fold {fold + 1} - Loss')

#     # Plotting the MSEs
#     plt.subplot(1, 3, 2)
#     plt.plot(epochs, train_mses, 'bo-', label='Training MSE')
#     plt.plot(epochs, val_mses, 'ro-', label='Validation MSE')
#     plt.xlabel('Epochs')
#     plt.ylabel('MSE')
#     plt.legend()
#     plt.title(f'{model_name} - Fold {fold + 1} - MSE')

#     # Plotting the R2 scores
#     plt.subplot(1, 3, 3)
#     plt.plot(epochs, train_r2s, 'bo-', label='Training R2')
#     plt.plot(epochs, val_r2s, 'ro-', label='Validation R2')
#     plt.xlabel('Epochs')
#     plt.ylabel('R2 Score')
#     plt.legend()
#     plt.title(f'{model_name} - Fold {fold + 1} - R2 Score')

#     plt.tight_layout()
#     plt.savefig(os.path.join(path, f'{model_name}_fold_{fold + 1}_performance.png'))
#     plt.close()

# def plot_loss_and_scatter(train_losses, val_losses, train_mses, val_mses, train_r2s, val_r2s, model_name, path, fold=None):
#     epochs = range(1, len(train_losses) + 1)
#     plt.figure(figsize=(15, 5))

#     # Plotting the losses
#     plt.subplot(1, 3, 1)
#     plt.plot(epochs, train_losses, 'bo-', label='Training loss')
#     plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     # plt.title(f'{model_name} - {"Fold " + str(fold + 1) if fold is not None else ""} - Loss')
#     plt.title(f'{model_name} - {f"Fold {fold + 1}" if fold is not None else ""} - Loss')

#     # Plotting the MSEs
#     plt.subplot(1, 3, 2)
#     plt.plot(epochs, train_mses, 'bo-', label='Training MSE')
#     plt.plot(epochs, val_mses, 'ro-', label='Validation MSE')
#     plt.xlabel('Epochs')
#     plt.ylabel('MSE')
#     plt.legend()
#     # plt.title(f'{model_name} - {"Fold " + str(fold + 1) if fold is not None else ""} - MSE')
#     plt.title(f'{model_name} - {f"Fold {fold + 1}" if fold is not None else ""} - MSE')

#     # Plotting the R2 scores
#     plt.subplot(1, 3, 3)
#     plt.plot(epochs, train_r2s, 'bo-', label='Training R2')
#     plt.plot(epochs, val_r2s, 'ro-', label='Validation R2')
#     plt.xlabel('Epochs')
#     plt.ylabel('R2 Score')
#     plt.legend()
#     # plt.title(f'{model_name} - {"Fold " + str(fold + 1) if fold is not None else ""} - R2 Score')
#     plt.title(f'{model_name} - {f"Fold {fold + 1}" if fold is not None else ""} - R2 Score')

#     plt.tight_layout()
#     # plt.savefig(os.path.join(path, f'{model_name}_{"fold_" + str(fold + 1) if fold is not None else "performance"}.png'))
#     plt.savefig(os.path.join(path, f'{model_name}_{f"fold_{fold + 1}" if fold is not None else "performance"}.png'))
#     plt.close()

def plot_loss_and_scatter(train_losses, val_losses, train_mses, val_mses, train_r2s, val_r2s, model_name, path, fold=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))

    # Plotting the losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - {"Fold " + str(fold) if fold is not None else ""} - Loss')

    # Plotting the MSEs
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_mses, 'bo-', label='Training MSE')
    plt.plot(epochs, val_mses, 'ro-', label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f'{model_name} - {"Fold " + str(fold) if fold is not None else ""} - MSE')

    # Plotting the R2 scores
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_r2s, 'bo-', label='Training R2')
    plt.plot(epochs, val_r2s, 'ro-', label='Validation R2')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.title(f'{model_name} - {"Fold " + str(fold) if fold is not None else ""} - R2 Score')

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{model_name}_{"fold_" + str(fold) if fold is not None else "performance"}.png'))
    plt.close()