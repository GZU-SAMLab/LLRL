import os
import sys
import json
import torch
from torchvision import transforms, datasets


current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

sys.path.append(project_root)

from model.train import Trainer  

def load_and_test_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), args.data_path))  
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)

   
    test_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=data_transform)
    test_num = len(test_dataset)

    
    class_to_idx = test_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_to_idx.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)

    print(f"Using {test_num} images for testing.")

    
    net = Trainer('small', 'classifier', args.num_class)
    net.to(device)

    
    model_weight_path = args.model_path  
    assert os.path.exists(model_weight_path), f"file {model_weight_path} does not exist."

    net.load_state_dict(torch.load(model_weight_path), strict=False)
    print("Loaded trained model weights.")

    
    correct = 0
    total = 0

   
    net.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            outputs = net(test_images.to(device), test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]

            
            total += test_labels.size(0)

            
            correct += (predict_y == test_labels.to(device)).sum().item()

    accuracy = correct / total * 100
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f}%")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--data-path', type=str, default="../leaf_diseases")  
    parser.add_argument('--model-path', type=str, default="./resultClassifier/model1.pth")  
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    load_and_test_model(args)
