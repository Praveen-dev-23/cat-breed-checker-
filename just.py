"""
Free Open-Source Image Generation
Uses Stable Diffusion via Hugging Face (Completely Free!)
"""
import torch
from diffusers import StableDiffusionPipeline
import os
from pathlib import Path
from datetime import datetime




# Create output directory
OUTPUT_DIR = "generated_images"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def generate_image(prompt, num_images=1, height=512, width=512):
    """
    Generate images using Stable Diffusion (Free & Open Source)
    
    Args:
        prompt: Text description of what you want to generate
        num_images: How many images to generate
        height: Image height (default 512)
        width: Image width (default 512)
    """
    try:
        print("\n" + "="*60)
        print("🎨 FREE IMAGE GENERATION WITH STABLE DIFFUSION")
        print("="*60)
        print(f"📝 Prompt: {prompt}")
        print(f"🖼️  Generating {num_images} image(s)...")
        
        # Check if GPU is available (faster)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Using device: {device.upper()}")
        
        if device == "cpu":
            print("⚠️  CPU mode (slower). GPU recommended: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        # Load model (first time downloads ~4GB, then cached locally)
        print("\n📥 Loading Stable Diffusion model (first run downloads ~4GB)...")
        model_id = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            safety_checker=None  # Disable safety checker for faster generation
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()  # For faster generation on limited memory
        
        print("✅ Model loaded!")
        
        # Generate images
        print(f"\n⏳ Generating images (this may take a minute or two)...")
        
        for i in range(num_images):
            print(f"\n   Generating image {i+1}/{num_images}...")
            
            image = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_DIR}/image_{timestamp}_{i+1}.png"
            image.save(filename)
            print(f"   ✅ Saved to: {filename}")
        
        print("\n" + "="*60)
        print(f"🎉 SUCCESS! Generated {num_images} image(s)")
        print(f"📁 Location: {OUTPUT_DIR}/")
        print("="*60 + "\n")

    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU out of memory. Try:")
        print("   1. Generate smaller images (256x256)")
        print("   2. Generate fewer images")
        print("   3. Use CPU instead")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTo install required packages, run:")
        print("   pip install torch diffusers transformers pillow")


if __name__ == "__main__":
    # Example prompts (modify these!)
    prompts = [
        "A beautiful sunset over mountains, oil painting style",
        "A futuristic city with flying cars and neon lights",
        "A cute golden retriever playing in the snow",
    ]
    
    print("\n" + "🌟 "*10)
    print("FREE & OPEN-SOURCE IMAGE GENERATION")
    print("Using Stable Diffusion v1.5")
    print("🌟 "*10)
    
    # Generate image from first prompt
    generate_image(
        prompt=prompts[0],
        num_images=1,
        height=512,
        width=512
    )
    
    # Uncomment to generate more images:
    # generate_image(prompts[1], num_images=1)
    # generate_image(prompts[2], num_images=1)
    
    # Interactive mode (uncomment to enable)
    """
    while True:
        user_prompt = input("\n🎨 Enter image description (or 'quit' to exit): ").strip()
        if user_prompt.lower() == 'quit':
            break
        generate_image(user_prompt, num_images=1)
    """
