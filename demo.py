import math

import numpy as np
import torch
from torchvision import transforms

from config import device
from utils import align_face, get_face_all_attributes

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
transformer = data_transforms['val']


def get_image(filename):
    _, _, landmarks = get_face_all_attributes(filename)
    img = align_face(filename, landmarks)
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.to(device)
    return img


if __name__ == "__main__":
    id_card_fn = 'id_card.jpg'
    img0 = get_image(id_card_fn)
    photo_fn = 'photo_1.jpg'
    img1 = get_image(photo_fn)
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
    imgs[0] = img0
    imgs[1] = img1

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    threshold = 73.18799151798612

    output = model(imgs)

    feature0 = output[0].cpu().numpy()
    feature1 = output[1].cpu().numpy()
    x0 = feature0 / np.linalg.norm(feature0)
    x1 = feature1 / np.linalg.norm(feature1)
    cosine = np.dot(x0, x1)
    theta = math.acos(cosine)
    theta = theta * 180 / math.pi

    print(theta)
    print(theta < threshold)
