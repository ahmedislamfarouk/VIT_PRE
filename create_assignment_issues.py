import argparse
from github import Github

ISSUES = [
    {
        "title": "RESEARCH - CLIP Architecture & Zero-Shot Capability Deep Dive",
        "body": """**Objective:** Understand the underlying mechanics of CLIP (Contrastive Language-Image Pre-Training) to inform prompt design.

**Subtasks:**
- Research the dual-encoder architecture (Image Encoder vs. Text Encoder).
- Analyze the contrastive loss function and how it enables zero-shot classification.
- Document how cosine similarity is used to rank text-image pairs.
- Identify the specific CLIP variants (e.g., ViT-B/32, RN50) and their trade-offs.

**Acceptance Criteria (Definition of Done):**
- Technical summary of CLIP's architecture and inference pipeline.
- Selection of CLIP model variant for the lab.

**Technical Details:**
- Focus on the multimodal embedding space.
- Review "Learning Transferable Visual Models From Natural Language Supervision".
""",
        "labels": ["Research"]
    },
    {
        "title": "SETUP - Environment & Dataset Curation",
        "body": """**Objective:** Set up the development environment and curate a diverse dataset of 8-12 images.

**Subtasks:**
- Install dependencies (torch, clip, PIL).
- Curate 10 images covering objects, scenes, and complex/crowded images.
- Organize images in a structured directory format.
- Create a script to load and preprocess images for CLIP.

**Acceptance Criteria (Definition of Done):**
- Repository with requirements.txt and data/ folder.
- Python script that successfully loads and displays curated images.

**Technical Details:**
- Image variations: Animals, vehicles, people, indoor, outdoor.
- Complexity levels: Simple vs. Crowded.
""",
        "labels": ["Setup", "Data"]
    },
    {
        "title": "DESIGN - Systematic Prompt Engineering Framework",
        "body": """**Objective:** Design a multi-category prompt framework for evaluating CLIP's semantic alignment.

**Subtasks:**
- For each image, define:
    - 2+ Basic prompts ("a dog").
    - 2+ Descriptive prompts ("a small brown dog running in a park").
    - 2+ Contextual prompts ("an animal playing outdoors in nature").
    - 2+ Misleading/Adversarial prompts ("a cat" for a dog image).
- Create a JSON mapping file connecting images to their designed prompts.
- Document the rationale behind the adversarial choices.

**Acceptance Criteria (Definition of Done):**
- prompts.json containing at least 80 prompts (10 images * 4 categories * 2 variations).
""",
        "labels": ["Design"]
    },
    {
        "title": "IMPLEMENTATION - CLIP Inference & Scoring Engine",
        "body": """**Objective:** Build a robust Python script to compute and rank similarity scores for all image-prompt pairs.

**Subtasks:**
- Implement CLIP inference loop.
- Calculate softmax-normalized similarity scores (confidence).
- Sort prompts by score per image.
- Export results to a structured format (CSV or JSON).

**Acceptance Criteria (Definition of Done):**
- inference.py script that produces a results.json.

**Technical Details:**
- Use torch.no_grad() for inference.
- Ensure proper normalization of embeddings before cosine similarity.
""",
        "labels": ["Implementation"]
    },
    {
        "title": "ANALYSIS - Performance Metrics & Prompt Sensitivity Analysis",
        "body": """**Objective:** Perform a deep technical analysis of the results.

**Subtasks:**
- Analyze the correlation between prompt detail and confidence scores.
- Evaluate the impact of prompt length vs. semantic density.
- Identify patterns in "failure cases" (e.g., when adversarial prompts outrank correct ones).
- Compare specific vs. general prompt performance.

**Acceptance Criteria (Definition of Done):**
- Comprehensive analysis report (Markdown) with data-backed insights.
""",
        "labels": ["Analysis"]
    },
    {
        "title": "OPTIMIZATION - Strategy for High-Fidelity Prompting",
        "body": """**Objective:** Propose and validate an optimized prompting strategy.

**Subtasks:**
- Define a "Before vs. After" comparison for 3 specific failure cases.
- Formalize a strategy for writing effective prompts (e.g., "The CLIP Prompting Protocol").
- Test the strategy on a new set of images.

**Acceptance Criteria (Definition of Done):**
- Documented strategy and validation results.
""",
        "labels": ["Optimization"]
    },
    {
        "title": "ADVANCED - Word Sensitivity & Structural Bias Investigation",
        "body": """**Objective:** Execute the advanced challenge to identify model biases or structural sensitivities.

**Subtasks:**
- Compare "photo of [X]" vs "image of [X]" vs "picture of [X]".
- Test short vs. long vs. structured prompts.
- Analyze any observed biases (e.g., preference for certain adjectives or nouns).

**Acceptance Criteria (Definition of Done):**
- Detailed findings in the final report.
""",
        "labels": ["Advanced"]
    }
]

def main():
    parser = argparse.ArgumentParser(description="Create assignment issues on GitHub.")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., user/repo)")
    args = parser.parse_args()

    gh = Github(args.token)
    repo = gh.get_repo(args.repo)
    
    print(f"Connecting to repository: {args.repo}...")
    
    for issue_data in ISSUES:
        try:
            issue = repo.create_issue(
                title=issue_data["title"],
                body=issue_data["body"],
                labels=issue_data["labels"]
            )
            print(f"  ✅ Created Issue #{issue.number}: {issue.title}")
        except Exception as e:
            print(f"  ❌ Failed to create issue '{issue_data['title']}': {e}")

    print("\nAll issues processed!")

if __name__ == "__main__":
    main()
