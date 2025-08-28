import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import torch
from diffusers import StableDiffusionPipeline

def main():
    """
    Generates an image from a text prompt using Stable Diffusion.
    """
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using Stable Diffusion.")
    parser.add_argument("--prompt", type=str, required=True, help="A detailed description of the image to generate.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Concepts to exclude from the image.")
    parser.add_argument("--output_path", type=str, default="generated_image.png", help="The path to save the generated image.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="The number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="How much the model should adhere to the prompt.")

    args = parser.parse_args()

    # --- 1. Load the Model ---
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    except Exception as e:
        print(f"Could not load the model: {e}")
        print("Please ensure you have an internet connection and the necessary libraries installed.")
        return

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Using GPU for generation.")
    else:
        print("GPU not available, using CPU. This will be slow.")

    # --- 2. Generate the Image ---
    print(f"Generating image for prompt: '{args.prompt}'")
    try:
        image = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        ).images[0]

        # --- 3. Save the Image ---
        image.save(args.output_path)
        print(f"Image saved as {args.output_path}")

    except Exception as e:
        print(f"An error occurred during image generation: {e}")

if __name__ == "__main__":
    main()
