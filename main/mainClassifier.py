import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets


current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

sys.path.append(project_root)

from model.train import Trainer  

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), args.data_path))  
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)

    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)

    test_dataset = datasets.ImageFolder(root=os.path.join(data_root, "test"),
                                        transform=data_transform["test"])
    test_num = len(test_dataset)

    
    class_to_idx = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_to_idx.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=nw)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)

    print("using {} images for training, {} images for validation, {} images for test.".format(train_num, val_num, test_num))

    
    net = Trainer('small', 'classifier', args.num_class)
    net.to(device)

    
    model_weight_path = args.model_path  
    assert os.path.exists(model_weight_path), f"file {model_weight_path} does not exist."

    
    state_dict = torch.load(model_weight_path, map_location=args.device)

    
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if not (k.startswith('encoder.resnet18.fc') or k.startswith('encoder.resnet.fc'))}

    
    net.load_state_dict(filtered_state_dict, strict=False)
    print("Loaded pre-trained weights, excluding encoder.resnet18.fc layer.")

    
    for name, param in net.encoder.named_parameters():
        param.requires_grad = False

    for name, param in net.decoder.structure_enhance.named_parameters():
        param.requires_grad = False


    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.01)

    epochs = args.epochs
    best_acc = 0.0
    save_path = args.save_path
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device), images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device), val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            print(f'best accurate is:{best_acc}')
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    net.load_state_dict(torch.load(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str,
                        default="../leaf_diseases")  # gai
    parser.add_argument('--model-path', type=str,
                        default="../weight")  # gai
    parser.add_argument('--save-path', type=str,
                        default="./resultClassifier/tomato_model.pth")  # gai
    parser.add_argument('--model-name', default='ResNet18', help='model name')

    parser.add_argument('--weights', type=str, default="", help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

