import torch
from PIL import Image
import os
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
ROOT_DIR = "data/liuyifei"   # image dir
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model, preprocess = load_from_name("RN50", \
                                    device=device,
                                    download_root="./")

model.eval()

# image preprocess
infer_data = None
image_files = os.listdir(ROOT_DIR)
for name in image_files:
    image_path = os.path.join(ROOT_DIR, name)
    image_data = Image.open(image_path)
    image_data = preprocess(image_data).unsqueeze(0)                         # [batch, 3, 224, 224]
    if infer_data is None:
        infer_data = image_data
    else:
        infer_data = torch.cat([infer_data, image_data], dim=0)
infer_data = infer_data.to(device)

# text data
text_data = clip.tokenize(["倪妮", "刘亦菲"]).to(device)        # [4, 64]

with torch.no_grad():
    image_features = model.encode_image(infer_data)     # [batch, 1024]
    # text_features  = model.encode_text(text_data)       # [text_samples, 1024]
    text_features  = model.encode_image(infer_data)

    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # logits_per_image 每张对象与每个text之间的相似度
    # cosine similarity as logits
    # logit_scale = self.logit_scale.exp()
    # logits_per_image = logit_scale * image_features @ text_features.t()
    # logits_per_text = logits_per_image.t()

    # logits_per_image, logits_per_text = model.get_similarity(infer_data, text_data)     
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
for idx, prob in enumerate(probs):
    print("file name: {} Label probs: {}".format(image_files[idx], prob) ) 