import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    cfg
):
    model.train()
    max_batches = cfg["train"].get("max_batches", None)
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if max_batches is not None and (batch_idx + 1) >= max_batches:
         break

    epoch_loss = running_loss / (batch_idx + 1)

    return epoch_loss