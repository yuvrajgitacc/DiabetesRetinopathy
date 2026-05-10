import torch
from engine.detector import _load_pytorch_model, _preprocess_for_effnet
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

# Load the newly trained model
model = _load_pytorch_model()
if model is None:
    print("Model not loaded")
    exit(1)
model.eval()

DATA_DIR = Path("data/aptos/colored_images")
STAGE_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}

total_correct = Counter()
total_images = Counter()

# Let's evaluate a sample of images from each class
print(f"{'='*60}\nEvaluating newly trained model (Stage-by-Stage)\n{'='*60}")

for stage in range(5):
    class_dir = DATA_DIR / str(stage)
    if not class_dir.exists():
        continue
        
    images = list(class_dir.glob("*.png"))[:50] # Test 50 images per stage
    
    for img_path in images:
        img = np.array(Image.open(img_path).convert("RGB"))
        preprocessed = _preprocess_for_effnet(img)
        tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).unsqueeze(0).float()
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        
        with torch.no_grad():
            output = model(tensor)
            pred = int(torch.argmax(output, dim=1).item())
            
        total_images[stage] += 1
        if pred == stage:
            total_correct[stage] += 1

print("\n[NEW MODEL] Stage-by-Stage Accuracy:")
overall_correct = 0
overall_total = 0
for stage in range(5):
    if total_images[stage] > 0:
        acc = (total_correct[stage] / total_images[stage]) * 100
        print(f"  Stage {stage} ({STAGE_NAMES[stage]}): {acc:.1f}% ({total_correct[stage]}/{total_images[stage]})")
        overall_correct += total_correct[stage]
        overall_total += total_images[stage]

print(f"\n[NEW MODEL] Overall Sample Accuracy: {(overall_correct/overall_total)*100:.1f}%\n")
