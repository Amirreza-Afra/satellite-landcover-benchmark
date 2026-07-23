import torch


def validate(
    model,
    dataloader,
    criterion,
    metrics,
    device,
):

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            acc = metrics["accuracy"](outputs, labels)

            running_loss += loss.item()
            running_acc += acc

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)

    return epoch_loss, epoch_acc