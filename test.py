import torch

def pgd_test(model, device, test_loader, criterion, pgd_adv):

    model.eval()
    test_pgd_iter = 20

    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            adv = pgd_adv.perturb(inputs, targets, test_pgd_iter)
            adv_outputs = model(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()
    
    benign_acc_log = 'Total benign test accuracy: ' + str(100. * benign_correct / total)
    adv_acc_log = 'Total adversarial test accuracy: ' + str(100. * adv_correct / total)
    benign_loss_log = 'Total benign test loss: ' + str(benign_loss)
    adv_loss_log = 'Total adversarial test loss: ' + str(adv_loss)

    print(benign_acc_log)
    print(adv_acc_log)
    print(benign_loss_log)
    print(adv_loss_log)