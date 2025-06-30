import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ✅ Load your local image
img_path = 'input.png'  # Place the file in the same folder
image = Image.open(img_path).convert('RGB')

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)

# Load model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)['out'][0]
pred = output.argmax(0).byte().cpu().numpy()

# Show result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(pred)
plt.axis('off')
plt.tight_layout()
plt.show()