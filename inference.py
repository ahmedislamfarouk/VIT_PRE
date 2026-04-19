import torch
import clip
from PIL import Image
import json
import os

# Define models to compare
MODELS_TO_COMPARE = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_inference():
    with open("prompts.json", "r") as f:
        prompts_data = json.load(f)
    
    all_model_results = {}
    
    for model_name in MODELS_TO_COMPARE:
        print(f"--- Loading model: {model_name} ---")
        model, preprocess = clip.load(model_name, device=device)
        results = {}
        
        for img_name, categories in prompts_data.items():
            img_path = os.path.join("data", img_name)
            if not os.path.exists(img_path):
                continue
                
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
            print(f"Model {model_name} processed {img_name}")
            
        all_model_results[model_name] = results

    with open("results.json", "w") as f:
        json.dump(all_model_results, f, indent=2)
    print("All results saved to results.json")

if __name__ == "__main__":
    run_inference()
