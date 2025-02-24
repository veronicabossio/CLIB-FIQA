# CLIB analysis module

import torch
import torch.nn.functional as F
from pathlib import Path
import shutil
from itertools import product
from PIL import Image
import torchvision.transforms as T

# Import model functions
from model import clip
from model.models import convert_weights
from utilities import dist_to_score


# Define Categories
quality_list = ['bad', 'poor', 'fair', 'good', 'perfect']
blur_list = ['hazy', 'blurry', 'clear']
occ_list = ['obstructed', 'unobstructed']
pose_list = ['profile', 'slight angle', 'frontal']
exp_list = ['exaggerated expression', 'typical expression']
ill_list = ['extreme lighting', 'normal lighting']

# Tokenized text prompts
joint_texts = torch.cat([
    clip.tokenize(f"a photo of a {b}, {o}, and {p} face displaying a {e} under {l}, which is of {q} quality")
    for b, o, p, e, l, q in product(blur_list, occ_list, pose_list, exp_list, ill_list, quality_list)
]).cuda()

# Category Maps
pose_map = {i: v for i, v in enumerate(pose_list)}
blur_map = {i: v for i, v in enumerate(blur_list)}
occ_map = {i: v for i, v in enumerate(occ_list)}
ill_map = {i: v for i, v in enumerate(ill_list)}
exp_map = {i: v for i, v in enumerate(exp_list)}

# Load Model
def load_clip_model():
    clip_model_path = "./weights/RN50.pt"
    clip_weights = "./weights/CLIB-FIQA_R50.pth"
    model = clip.load(clip_model_path, device='cuda', jit=False)[0]
    model = convert_weights(model)
    model.load_state_dict(torch.load(clip_weights, map_location="cuda"))
    model.eval()
    return model

# Preprocessing Function
def img_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return transform(img).unsqueeze(0)

model = load_clip_model()
# Inference Function
@torch.no_grad()
def analyze_image(image_path):
    tensor_data = img_tensor(image_path).cuda()
    logits_per_image, _ = model(tensor_data, joint_texts)
    
    logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))

    logits_quality = logits_per_image.sum((1, 1, 1, 1, 1))
    logits_blur = torch.argmax(logits_per_image.sum((6, 5, 4, 3, 2)), dim=1).cpu().numpy().squeeze(0)
    logits_occ = torch.argmax(logits_per_image.sum((6, 5, 4, 3, 1)), dim=1).cpu().numpy().squeeze(0)
    logits_pose = torch.argmax(logits_per_image.sum((6, 5, 4, 2, 1)), dim=1).cpu().numpy().squeeze(0)
    logits_exp = torch.argmax(logits_per_image.sum((6, 5, 3, 2, 1)), dim=1).cpu().numpy().squeeze(0)
    logits_ill = torch.argmax(logits_per_image.sum((6, 4, 3, 2, 1)), dim=1).cpu().numpy().squeeze(0)
    
    quality_preds = dist_to_score(logits_quality).cpu().numpy().squeeze(0)

    # Get expression probability
    logits_exp_values = logits_per_image.sum((6, 5, 3, 2, 1))
    expression_probs = torch.softmax(logits_exp_values, dim=1).cpu().numpy().squeeze(0)
    exaggeration_score = expression_probs[0]  # Probability of 'exaggerated expression'

    # Create summary message
    output_msg = f"A photo of a [{blur_map[int(logits_blur)]}], [{occ_map[int(logits_occ)]}], and [{pose_map[int(logits_pose)]}] face with [{exp_map[int(logits_exp)]}] under [{ill_map[int(logits_ill)]}]"
    
    return {
        "image_path": str(image_path),
        "quality": quality_preds,
        "pose": pose_map[int(logits_pose)],
        "expression_score": exaggeration_score,
        "expression": exp_map[int(logits_exp)],
        "message": output_msg
    }

# Batch Filtering Function
def batch_filter_images(input_dir, output_dir, model, quality_thresh=None, pose_filter=None, expression_thresh=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in input_dir.iterdir():
        if not image_path.is_file():
            continue

        results = analyze_image(image_path, model)

        # Apply filtering conditions
        if (quality_thresh is not None and results["quality"] < quality_thresh):
            continue
        if (pose_filter is not None and results["pose"] != pose_filter):
            continue
        if (expression_thresh is not None and results["expression_score"] > expression_thresh):
            continue

        # Copy the filtered image
        shutil.copy(image_path, output_dir / image_path.name)
        print(f"Copied: {image_path.name} - {results['message']}")

    print(f"Filtering complete. Processed images saved in {output_dir}")

# Example Usage
# if __name__ == "__main__":
#     model = load_clip_model()

#     # **Test a single image**
#     image_summary = analyze_image("/path/to/face.jpg", model)
#     print(image_summary)

#     # **Batch filtering with multiple conditions**
#     batch_filter_images(
#         input_dir="/home/veronicabossio/face_datasets/filter_test_neighbors",
#         output_dir="/home/veronicabossio/face_datasets/filtered_faces",
#         model=model,
#         quality_thresh=0.78,    # Only keep images with quality > 0.78
#         pose_filter="frontal",  # Only keep frontal images
#         expression_thresh=0.3   # Remove exaggerated expressions above 0.3 probability
#     )
