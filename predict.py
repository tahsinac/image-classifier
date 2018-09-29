import argparse
import json
import numpy as np
import torch
from PIL import Image

def main():
    args = parse_args()
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
    model = load_checkpoint(args.checkpoint)
    
    if args.gpu:
        model = model.to('cuda')
            
    probs, classes = predict(args.input_image_path, model, args.topk)
    labels = []
    for k, v in cat_to_name.items():
        if k in classes:
            labels.append(v)
    print(labels)
    print(probs)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
  
    # Resizing
    image_ratio = image.size[1] / image.size[0]
    if image_ratio > 1:
        image = image.resize((256, int(image_ratio*256)))
    else:
        image = image.resize(int(image_ratio*256), 256)
    width, height = image.size
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    image = image.crop((left, top, right, bottom))
   
    # Converting color channels from 0-255 to 0-1
    image = np.array(image)
    image = image/255
    
    # Normalizing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Reordering
    image = image.transpose((2, 0, 1))
    
    return image
            
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.to(device)
    image = image.unsqueeze(0)
    output = model.forward(image)
    ps = torch.exp(output)
    prob, indx = ps.topk(topk)
    prob_list = prob.tolist()[0]
    indx_list = indx.tolist()[0]
    #Inverting class_to_idx dictionary, as we now have the indices
    idx_to_class = {y:x for x,y in model.class_to_idx.items()}
    class_list = []
    for k, v in idx_to_class.items():
        if k in indx_list:
            class_list.append(v)
        
    return prob_list, class_list
                

def parse_args():
    parser = argparse.ArgumentParser(description = 'This loads a model from a checkpoint and makes a prediction on an image.')
    parser.add_argument('--input_image_path', type = str, default = '/home/workspace/aipnd-project/flowers/test/100/image_07939.jpg', help ='Image path to image for prediction')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Model checkpoint')
    parser.add_argument('--gpu', type = bool, default = False, help = 'Use GPU for training if available')
    parser.add_argument('--topk', type = int, default = 5, help = 'TopK value')
    parser.add_argument('--categories', action = 'store', default = 'cat_to_name.json', help = ".json file containing names of classes")
    return parser.parse_args()


main()
