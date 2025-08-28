import argparse
from diffusers import OnnxStableDiffusionPipeline, DPMSolverMultistepScheduler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--output_path", default="generated_image.png")
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=6.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    args = p.parse_args()

    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        "sd15-onnx",                     # <-- exported folder
        provider="DmlExecutionProvider", # DirectML for AMD iGPU
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height, width=args.width,
    ).images[0]
    image.save(args.output_path)
    print("Saved:", args.output_path)

if __name__ == "__main__":
    main()
