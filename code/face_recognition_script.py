import torch
from torchvision import transforms
from PIL import Image
import os
from backbone import get_model  
import model



os.chdir("C:\\Users\\mathe\\StudioProjects\\tfm_viu\\src\\assets")



model =  torch.load("KD_full_mobiFace_like_nologits_v1_4.pth") 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])


image = Image.open("me.jpg")
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)


model.eval()
embeddings1 = model(image_tensor)


image2 = Image.open("me3.jpg")
image_tensor2 = transform(image2)
image_tensor2 = image_tensor2.unsqueeze(0)
image_tensor2 = image_tensor2.to(device)

embeddings2 = model(image_tensor2)


def cosine_similarity(embeddings1, embeddings2):
    dot_product = torch.dot(embeddings1, embeddings2)
    norm_embeddings1 = torch.norm(embeddings1)
    norm_embeddings2 = torch.norm(embeddings2)
    return dot_product / (norm_embeddings1 * norm_embeddings2)


similarity =cosine_similarity(embeddings1[0], embeddings2[0])

if similarity.item() >0.2:
    result =1
else:
    result=0

print(result)
