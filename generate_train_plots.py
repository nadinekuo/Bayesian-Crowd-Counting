import matplotlib.pyplot as plt

def train_array_from_txt(txt_path):
    loss, mse, mae = [], [], []
    with open(txt_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            temp_line = line.strip().split(" ")
            if len(temp_line) > 4 and temp_line[4] == "Train,":
                loss.append(float(temp_line[6].replace(",", "")))
                mse.append(float(temp_line[8].replace(",", "")))
                mae.append(float(temp_line[10].replace(",", "")))

    return loss, mse, mae


def val_array_from_txt(txt_path):
    mse, mae = [], []
    with open(txt_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            temp_line = line.strip().split(" ")
            print(temp_line)
            if len(temp_line) > 4 and temp_line[4] == "Val,":
                mse.append(float(temp_line[6].replace(",", "")))
                mae.append(float(temp_line[8].replace(",", "")))

    return mse, mae

def generate_graphs(loss, train_mse, train_mae, val_mse, val_mae):
    epochs = range(1, len(loss) + 1)
    val_epochs = range(5, len(val_mse) * 5 + 1, 5)

    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot MSE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_mse, label='Training MSE')
    plt.plot(val_epochs, val_mse, label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_mae, label='Training MAE')
    plt.plot(val_epochs, val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_path = "vgg-train-IOCfish/bayplus/train.log.txt"
    loss, train_mse, train_mae = train_array_from_txt(log_path)
    val_mse, val_mae = val_array_from_txt(log_path)

    generate_graphs(loss, train_mse, train_mae, val_mse, val_mae)