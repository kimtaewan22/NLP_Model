import torch

def evaluate(model, test_loader):

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total