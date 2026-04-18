import torch
import clip
from PIL import Image
import json
import os

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def run_inference():
    with open("prompts.json", "r") as f:
        prompts_data = json.load(f)
    
    results = {}
    
    for img_name, categories in prompts_data.items():
        img_path = os.path.join("data", img_name)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        # Flatten prompts
        all_prompts = []
        prompt_map = []
        for cat, p_list in categories.items():
            for p in p_list:
                all_prompts.append(p)
                prompt_map.append({"text": p, "category": cat})
        
        text = clip.tokenize(all_prompts).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(len(all_prompts))
        
        img_results = []
        for value, index in zip(values, indices):
            idx = index.item()
            img_results.append({
                "prompt": prompt_map[idx]["text"],
                "category": prompt_map[idx]["category"],
                "score": float(value.item())
            })
        
        results[img_name] = img_results
        print(f"Processed {img_name}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results.json")

if __name__ == "__main__":
    run_inference()
