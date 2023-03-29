
import torch
import sys
from tqdm import tqdm


class Valid:
    def valid_method(self, model, device, test_loader, epoch):

        # validate
        model.eval()
        model = model.to(device)

        total = 0
        correct = 0.0

        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)    # file=sys.stdout 输出到控制台
            for step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                predict = torch.argmax(outputs, dim=1)
                total += val_labels.size(0)
                correct += torch.eq(predict, val_labels).sum().item()
                val_bar.desc = 'Valid Epoch {}, Acc {:.3f}%'.format(epoch, 100 * (correct / total))

        return correct / total
