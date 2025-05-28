import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def compare_images(original_img_path, clean_imgs, robust_imgs, clean_prompts=None, 
                  robust_prompts=None, save_path=None, return_fig=False):
    """
    Create a visualization comparing original image with its diffused versions.
    
    Args:
        original_img_path: Path to the original image
        clean_imgs: List of paths to clean diffused images
        robust_imgs: List of paths to robust diffused images
        clean_prompts: Dictionary mapping image IDs to clean prompts
        robust_prompts: Dictionary mapping image IDs to robust prompts
        save_path: Optional path to save the visualization
        return_fig: If True, return the figure object instead of showing/saving
    
    Returns:
        The figure object if return_fig is True, None otherwise
    """
    # Load all images
    original_img = Image.open(original_img_path)
    clean_images = [Image.open(img_path) for img_path in clean_imgs]
    robust_images = [Image.open(img_path) for img_path in robust_imgs]
    
    # Extract image ID from path
    img_name = os.path.basename(original_img_path)
    img_id = os.path.splitext(img_name)[0]
    
    # Create figure with 3 rows (original, clean diffusions, robust diffusions)
    # We'll need more height for the text, so adjust figsize
    fig, axs = plt.subplots(3, 5, figsize=(20, 15))
    
    # Add image name as figure title
    fig.suptitle(f"Comparison for {img_name}", fontsize=16)
    
    # Display original image in first row, first column
    axs[0, 0].imshow(np.array(original_img))
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")
    
    # Make the rest of the first row empty
    for i in range(1, 5):
        axs[0, i].axis("off")
    
    # Display clean diffusions in second row
    axs[1, 0].text(0.5, 0.5, "Clean\nDiffusions", ha="center", va="center", fontsize=12)
    axs[1, 0].axis("off")
    
    # Get the clean prompt if available
    clean_prompt = clean_prompts.get(img_name, "No prompt available") if clean_prompts else "No prompt available"
    
    for i, img in enumerate(clean_images):
        axs[1, i+1].imshow(np.array(img))
        axs[1, i+1].set_title(f"Clean {i+1}")
        axs[1, i+1].axis("off")
        
        # Add prompt text below the image
        if i == 0 and clean_prompt:  # Only show prompt once per row
            # Create a text box below the row for the prompt
            prompt_text = f"Clean Prompt: {clean_prompt}"
            fig.text(0.01, 0.6, prompt_text, ha='left', va='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                     wrap=True, fontsize=10)
    
    # Display robust diffusions in third row
    axs[2, 0].text(0.5, 0.5, "Robust\nDiffusions", ha="center", va="center", fontsize=12)
    axs[2, 0].axis("off")
    
    # Get the robust prompt if available
    robust_prompt = robust_prompts.get(img_name, "No prompt available") if robust_prompts else "No prompt available"
    
    for i, img in enumerate(robust_images):
        axs[2, i+1].imshow(np.array(img))
        axs[2, i+1].set_title(f"Robust {i+1}")
        axs[2, i+1].axis("off")
        
        # Add prompt text below the image
        if i == 0 and robust_prompt:  # Only show prompt once per row
            # Create a text box below the row for the prompt
            prompt_text = f"Robust Prompt: {robust_prompt}"
            fig.text(0.01, 0.3, prompt_text, ha='left', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                     wrap=False, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to make room for suptitle and prompts
    
    if return_fig:
        return fig
    elif save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    model_name = "vit-l"
    imgs_original_dir = "/mnt/datasets/coco/val2017/"

    if model_name == "vit-l":
        reconstructions_clean_file = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                            "results-coco-img-100smpls-3000iters-ViT-L-14-quickgelu-clean.json")
        reconstructions_robust_file = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                            "results-coco-img-100smpls-3000iters-ViT-L-14-quickgelu-robust.json")
        imgs_diffused_clean_dir = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                       "images-coco-clean/")
        imgs_diffused_robust_dir = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                          "images-coco-robust/")
        output_dir = "/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu"
    elif model_name == "vit-h":
        reconstructions_clean_file = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-H-14"
                                "results-coco-img-100smpls-3000iters-ViT-H-14-clean.json")
        reconstructions_robust_file = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-H-14"
                                "results-coco-img-100smpls-3000iters-ViT-H-14-robust.json")
        imgs_diffused_clean_dir = "/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-H-14/images-coco-clean/"
        imgs_diffused_robust_dir = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-H-14"
                                    "images-coco-inv-robust-gen-robust/")
        output_dir = "/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-H-14"


    # Create output directory for saving plots if needed

    os.makedirs(output_dir, exist_ok=True)

    # load reconstructed texts
    with open(reconstructions_clean_file, "r") as f:
        reconstructions_clean = json.load(f)
    with open(reconstructions_robust_file, "r") as f:
        reconstructions_robust = json.load(f)
    img_ids_prompts_clean = {
        el["original"]: el["reconstructed"] for el in reconstructions_clean["results"]
    }
    img_ids_prompts_robust = {
        el["original"]: el["reconstructed"] for el in reconstructions_robust["results"]
    }

    # get unique original img names
    img_names = [
        img_path.split("-")[1].split("-")[0] for img_path in os.listdir(imgs_diffused_clean_dir)
        if img_path.endswith(".png")
    ]
    img_names = sorted(list(set(img_names)), key=lambda x: int(x.split(".")[0]))
    img_names = img_names[:10]  # Limit to first 10 images for demonstration

    # Create a single PDF file for all comparisons
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = os.path.join(output_dir, "all_comparisons.pdf")
    
    with PdfPages(pdf_path) as pdf:
        # Process each image
        for img_name in img_names:
            print(f"Processing image: {img_name}")
            # Get the corresponding original image
            orig_img_path = os.path.join(imgs_original_dir, img_name)
            # Get image ID (filename without extension)
            img_id = os.path.splitext(img_name)[0]
            # Get the corresponding diffused images
            clean_imgs = sorted(glob.glob(os.path.join(imgs_diffused_clean_dir, f"*{img_name}*.png")))
            robust_imgs = sorted(glob.glob(os.path.join(imgs_diffused_robust_dir, f"*{img_name}*.png")))
            
            if len(clean_imgs) == 4 and len(robust_imgs) == 4:
                # Create comparison plot and add to PDF
                fig = compare_images(
                    orig_img_path, 
                    clean_imgs, 
                    robust_imgs, 
                    clean_prompts=img_ids_prompts_clean,
                    robust_prompts=img_ids_prompts_robust,
                    return_fig=True
                )
                pdf.savefig(fig)
                plt.close(fig)
            else:
                print(f"Skipping {img_name} - missing diffused images")
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Image Diffusion Comparison'
        d['Subject'] = 'Comparison between original, clean diffused, and robust diffused images'
        
    print(f"All comparisons saved to {pdf_path}")