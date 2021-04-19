def predict(model_path, image_path):
    from torch import nn
    import torch
#     import cv2
    import numpy as np
    from skimage import io, transform
    index_to_name = {0:'Covid', 1:'Normal', 2:'Viral Pnuemonia'}
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    image = io.imread(image_path)
    image = np.float32(transform.resize(image, (224, 224))) / 255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  
    model_eval = torch.load(model_path, map_location=device)
    image = torch.tensor(image)
    with torch.no_grad():
        image = image.to(device)
        model_eval = model_eval.to(device)
        model_eval.eval()
        outputs = model_eval(image)
        score, predicted = torch.max(outputs.data, 1)
    print(index_to_name[predicted.tolist()[0]])
    return index_to_name[predicted.tolist()[0]]
