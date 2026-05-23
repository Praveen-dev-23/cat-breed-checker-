#!/usr/bin/env python3
"""
🐱 Cat Breed Classifier
Identify cat breeds from images using Hugging Face (dima806/cat_breed_image_detection)
Featuring a highly refined and premium Gradio Web UI.
"""

import sys
import os
import subprocess
from pathlib import Path

# --- Dependency Autoloader ---
REQUIRED_PACKAGES = ["gradio", "transformers", "torch", "torchvision", "pillow"]

def check_dependencies():
    missing_packages = []
    for pkg in REQUIRED_PACKAGES:
        try:
            if pkg == "pillow":
                import PIL
            else:
                __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)
    
    if missing_packages:
        print("🔍 Detecting missing dependencies...")
        print(f"Installing {', '.join(missing_packages)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("✅ Dependencies successfully installed!")
        except Exception as e:
            print(f"❌ Autoinstallation failed: {e}")
            print(f"Please install dependencies manually using:\n  pip install {' '.join(missing_packages)}")
            sys.exit(1)

# Check and install dependencies if needed before loading ML modules
check_dependencies()

# Import ML packages after validation
import torch
from PIL import Image
from transformers import pipeline
import gradio as gr

# --- Cat Breed Facts Database (48 Breeds) ---
BREED_DATABASE = {
    "Abyssinian": {
        "origin": "Ethiopia (Abyssinia)",
        "temperament": "Active, playful, highly intelligent, curious, extroverted",
        "lifespan": "12-15 years",
        "fun_fact": "Abyssinian cats are known as the 'Clowns of the Cat Kingdom' because of their playful and mischievous nature."
    },
    "American Bobtail": {
        "origin": "United States",
        "temperament": "Loving, friendly, intelligent, adaptable, dog-like",
        "lifespan": "11-15 years",
        "fun_fact": "American Bobtails are known for their short 'bobbed' tail, which is a result of a natural genetic mutation."
    },
    "American Curl": {
        "origin": "United States",
        "temperament": "Affectionate, energetic, social, sweet-tempered, alert",
        "lifespan": "12-16 years",
        "fun_fact": "Their ears start straight and begin to curl backward in a graceful arc within a few days of birth."
    },
    "American Shorthair": {
        "origin": "United States",
        "temperament": "Gentle, calm, easygoing, friendly, quiet",
        "lifespan": "15-20 years",
        "fun_fact": "They were originally kept on ships to protect cargo from mice and rats during early journeys to America."
    },
    "Applehead Siamese": {
        "origin": "Thailand (Siam)",
        "temperament": "Vocal, affectionate, social, loyal, demanding",
        "lifespan": "15-20 years",
        "fun_fact": "They represent the traditional, rounder look of the Siamese breed, which was highly favored in the late 19th century."
    },
    "Balinese": {
        "origin": "United States / Thailand",
        "temperament": "Vocal, intelligent, playful, highly affectionate, graceful",
        "lifespan": "12-20 years",
        "fun_fact": "Named after the graceful dancers of Bali because of their elegant, long-haired build and fluid movements."
    },
    "Bengal": {
        "origin": "United States",
        "temperament": "Highly energetic, confident, curious, intelligent, wild-looking",
        "lifespan": "12-16 years",
        "fun_fact": "Bengals are a cross between domestic cats and the Asian Leopard Cat, giving them their beautiful spotted rosette coat."
    },
    "Birman": {
        "origin": "Myanmar (Burma)",
        "temperament": "Sweet, gentle, quiet, loving, easygoing",
        "lifespan": "12-16 years",
        "fun_fact": "Legends say that their striking white paws were blessed by a golden goddess as a symbol of purity."
    },
    "Bombay": {
        "origin": "United States",
        "temperament": "Outgoing, friendly, playful, highly affectionate, smart",
        "lifespan": "12-16 years",
        "fun_fact": "Bombay cats were bred to look like mini black panthers, featuring sleek black coats and copper-gold eyes."
    },
    "British Shorthair": {
        "origin": "United Kingdom",
        "temperament": "Calm, easygoing, independent, affectionate, quiet",
        "lifespan": "12-17 years",
        "fun_fact": "The British Shorthair is believed to be the inspiration behind the Cheshire Cat in Alice in Wonderland."
    },
    "Burmese": {
        "origin": "Thailand / Myanmar",
        "temperament": "Playful, vocal, social, highly affectionate, trustful",
        "lifespan": "10-16 years",
        "fun_fact": "Burmese cats maintain a kitten-like playfulness well into their senior years and form deep bonds with humans."
    },
    "Calico": {
        "origin": "Egypt / Global",
        "temperament": "Spirited, affectionate, energetic, independent, protective",
        "lifespan": "12-16 years",
        "fun_fact": "Calico is not a breed, but a color pattern that is almost exclusively female due to genetics (requiring two X chromosomes)."
    },
    "Cornish Rex": {
        "origin": "United Kingdom",
        "temperament": "Highly active, playful, intelligent, social, acrobatic",
        "lifespan": "11-15 years",
        "fun_fact": "They have extremely soft, wavy coats and large ears that make them look like adorable bat-eared elves."
    },
    "Devon Rex": {
        "origin": "United Kingdom",
        "temperament": "Mischievous, playful, loving, active, cuddly",
        "lifespan": "10-15 years",
        "fun_fact": "Devon Rexes have a distinct 'pixie-like' face and love to perch on their owner's shoulders."
    },
    "Dilute Calico": {
        "origin": "Global",
        "temperament": "Gentle, sweet, playful, loyal, loving",
        "lifespan": "12-16 years",
        "fun_fact": "A dilute calico displays the same tricolor pattern but with softer shades of grey, cream, and white."
    },
    "Dilute Tortoiseshell": {
        "origin": "Global",
        "temperament": "Independent, feisty, affectionate, talkative, loyal",
        "lifespan": "12-16 years",
        "fun_fact": "Often nicknamed 'Dilute Torties,' their mottled blue-grey and cream coats come with a fiery, independent personality."
    },
    "Domestic Long Hair": {
        "origin": "Global",
        "temperament": "Varied, adaptable, sturdy, friendly",
        "lifespan": "12-18 years",
        "fun_fact": "Unlike registered breeds, these are mixed-breed cats with long fur, representing a diverse and healthy genetic pool."
    },
    "Domestic Medium Hair": {
        "origin": "Global",
        "temperament": "Adaptable, friendly, easygoing, curious",
        "lifespan": "12-18 years",
        "fun_fact": "They possess a double-layered medium-length coat that is very soft, representing the most common mixed-breed cats."
    },
    "Domestic Short Hair": {
        "origin": "Global",
        "temperament": "Highly adaptable, friendly, resilient, playful, robust",
        "lifespan": "12-20 years",
        "fun_fact": "These are the ultimate companion cats, representing about 90-95% of all domestic cats in the world."
    },
    "Egyptian Mau": {
        "origin": "Egypt",
        "temperament": "Active, intelligent, loyal, playful, fast",
        "lifespan": "12-15 years",
        "fun_fact": "The Egyptian Mau is the fastest of all domestic cats, capable of reaching running speeds of up to 30 mph (48 km/h)."
    },
    "Exotic Shorthair": {
        "origin": "United States",
        "temperament": "Quiet, sweet, gentle, calm, affectionate",
        "lifespan": "12-15 years",
        "fun_fact": "Bred as a short-haired version of the Persian, they are often referred to as 'Persians in pajamas.'"
    },
    "Extra-Toes Cat - Hemingway Polydactyl": {
        "origin": "Key West, Florida (famed)",
        "temperament": "Friendly, outgoing, curious, intelligent, playful",
        "lifespan": "12-15 years",
        "fun_fact": "Sailors believed polydactyl cats (cats with extra toes) were lucky, and Ernest Hemingway's home in Key West is still famous for them."
    },
    "Havana": {
        "origin": "United Kingdom",
        "temperament": "Playful, outgoing, affectionate, intelligent, demanding",
        "lifespan": "10-15 years",
        "fun_fact": "They are named 'Havana Brown' because their rich chocolate-brown coat color closely matches the color of Havana cigars."
    },
    "Himalayan": {
        "origin": "United States / United Kingdom",
        "temperament": "Sweet, quiet, gentle, affectionate, placid",
        "lifespan": "12-15 years",
        "fun_fact": "Himalayans are a cross between Persians and Siamese, combining the fluffy coat of a Persian with the color points of a Siamese."
    },
    "Japanese Bobtail": {
        "origin": "Japan",
        "temperament": "Active, smart, vocal, friendly, energetic",
        "lifespan": "15-18 years",
        "fun_fact": "The Japanese Bobtail is the inspiration behind the iconic Japanese 'Maneki-neko' (beckoning lucky cat) figurine."
    },
    "Maine Coon": {
        "origin": "United States (Maine)",
        "temperament": "Friendly, gentle giant, playful, intelligent, social",
        "lifespan": "12-15 years",
        "fun_fact": "Maine Coons are one of the largest domestic cat breeds and have water-resistant coats and tufted paws to walk on snow."
    },
    "Manx": {
        "origin": "Isle of Man",
        "temperament": "Intelligent, playful, gentle, dog-like, protective",
        "lifespan": "10-14 years",
        "fun_fact": "The Manx cat is famous for being completely tailless ('rumpy') or having just a tiny stump ('stumpy')."
    },
    "Munchkin": {
        "origin": "United States",
        "temperament": "Energetic, outgoing, playful, sweet, adventurous",
        "lifespan": "12-15 years",
        "fun_fact": "Munchkins are characterized by their very short legs, caused by a natural genetic mutation, but they can run and jump surprisingly well."
    },
    "Nebelung": {
        "origin": "United States",
        "temperament": "Quiet, shy with strangers, loyal, affectionate, gentle",
        "lifespan": "13-16 years",
        "fun_fact": "The name Nebelung means 'creature of the mist' in German, describing their beautiful, shimmering blue-grey coat."
    },
    "Norwegian Forest": {
        "origin": "Norway",
        "temperament": "Independent, friendly, intelligent, calm, rugged",
        "lifespan": "14-16 years",
        "fun_fact": "Known as 'Wegies,' these cats feature in Norse mythology as giant cats that pulled the goddess Freya's chariot."
    },
    "Oriental Short Hair": {
        "origin": "United States / United Kingdom",
        "temperament": "Vocal, demanding, highly active, intelligent, social",
        "lifespan": "12-15 years",
        "fun_fact": "Closely related to the Siamese, they have large, bat-like ears and are available in over 300 color combinations."
    },
    "Persian": {
        "origin": "Iran (Persia)",
        "temperament": "Calm, quiet, sweet, dignified, gentle",
        "lifespan": "12-17 years",
        "fun_fact": "Persians are famous for their luxurious long coats, flat faces ('pansy faces'), and gentle, laid-back personalities."
    },
    "Ragamuffin": {
        "origin": "United States",
        "temperament": "Sweet, gentle, cuddle-loving, calm, sweet-tempered",
        "lifespan": "12-18 years",
        "fun_fact": "They are famous for their sweet expressions, thick rabbit-like fur, and the habit of going completely limp in your arms when held."
    },
    "Ragdoll": {
        "origin": "United States",
        "temperament": "Affectionate, placid, gentle, friendly, docile",
        "lifespan": "12-15 years",
        "fun_fact": "Ragdolls are named for their tendency to go limp and relaxed like a ragdoll when picked up."
    },
    "Russian Blue": {
        "origin": "Russia",
        "temperament": "Reserved, gentle, quiet, intelligent, loyal",
        "lifespan": "15-20 years",
        "fun_fact": "Russian Blues have a double coat that stands at a 45-degree angle, giving them a distinct shimmering, silvery appearance."
    },
    "Scottish Fold": {
        "origin": "Scotland",
        "temperament": "Sweet, adaptable, calm, loving, smart",
        "lifespan": "11-15 years",
        "fun_fact": "Their unique folded ears are caused by an incomplete dominant gene that affects the cartilage in their body."
    },
    "Siamese": {
        "origin": "Thailand (Siam)",
        "temperament": "Vocal, highly social, energetic, affectionate, talkative",
        "lifespan": "15-20 years",
        "fun_fact": "Siamese cats are born completely white and develop their dark color points (ears, face, paws) as they grow, due to temperature-sensitive enzymes."
    },
    "Siberian": {
        "origin": "Russia",
        "temperament": "Playful, adventurous, agile, loving, robust",
        "lifespan": "12-15 years",
        "fun_fact": "Siberian cats have a triple-layered water-repellent coat and produce lower levels of the Fel d 1 protein, making them more hypoallergenic."
    },
    "Snowshoe": {
        "origin": "United States",
        "temperament": "Friendly, vocal, playful, social, sweet",
        "lifespan": "12-15 years",
        "fun_fact": "Snowshoes are named for their bright white paws, which contrast beautifully with their darker pointed coats."
    },
    "Sphynx": {
        "origin": "Canada",
        "temperament": "Energetic, extroverted, silly, affectionate, warm",
        "lifespan": "12-15 years",
        "fun_fact": "Despite looking hairless, Sphynx cats are actually covered in a fine peach fuzz that makes them feel like warm suede."
    },
    "Tabby": {
        "origin": "Global",
        "temperament": "Outgoing, friendly, playful, intelligent, curious",
        "lifespan": "12-20 years",
        "fun_fact": "Tabby is not a breed but a coat pattern characterized by a distinct 'M' shape on the forehead, dating back to ancient Egypt."
    },
    "Tiger": {
        "origin": "Global",
        "temperament": "Active, wild-looking, brave, energetic, fierce hunter",
        "lifespan": "12-18 years",
        "fun_fact": "Often used to describe mixed-breed cats with vertical, tiger-like stripes (mackerel tabbies) that mimic their wild cousins."
    },
    "Tonkinese": {
        "origin": "Canada / United States",
        "temperament": "Playful, active, highly social, loving, vocal",
        "lifespan": "12-16 years",
        "fun_fact": "A cross between the Siamese and Burmese, they possess beautiful aqua-colored eyes and a lively, gregarious personality."
    },
    "Torbie": {
        "origin": "Global",
        "temperament": "Spunky, vocal, independent, affectionate, colorful",
        "lifespan": "12-20 years",
        "fun_fact": "Torbie is short for 'tortoiseshell-tabby,' presenting stripes mixed with patches of orange, brown, and cream."
    },
    "Tortoiseshell": {
        "origin": "Global",
        "temperament": "Feisty, independent, loyal, energetic, vocal",
        "lifespan": "12-20 years",
        "fun_fact": "Famous for 'tortitude'—the sassy and bold attitude that owners swear is linked to their orange and black patched coats."
    },
    "Turkish Angora": {
        "origin": "Turkey",
        "temperament": "Smart, playful, energetic, affectionate, loyal",
        "lifespan": "12-18 years",
        "fun_fact": "Highly revered in Turkey, many Turkish Angoras have heterochromia (one blue eye and one green/amber eye)."
    },
    "Turkish Van": {
        "origin": "Turkey",
        "temperament": "Active, intelligent, curious, water-loving, athletic",
        "lifespan": "12-17 years",
        "fun_fact": "Often called the 'swimming cat,' Turkish Vans have a unique water-resistant coat and love playing in water."
    },
    "Tuxedo": {
        "origin": "Global",
        "temperament": "Friendly, smart, energetic, vocal, charismatic",
        "lifespan": "12-20 years",
        "fun_fact": "Tuxedo cats are bicolor cats dressed to impress, and historical figures like Sir Isaac Newton and Shakespeare owned them."
    }
}

# --- Load Hugging Face Pipeline ---
print("\n" + "="*50)
print("🚀 STARTING CAT BREED CLASSIFIER APPLICATION")
print("="*50)
print("📥 Loading Model 'dima806/cat_breed_image_detection'...")
print("⚠️  (On first run, this downloads a ~340MB Vision Transformer model)\n")

# Use Apple MPS, Nvidia CUDA, or CPU device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Running ML Pipeline on device: {device.upper()}")

try:
    classifier = pipeline("image-classification", model="dima806/cat_breed_image_detection", device=device)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Device {device} failed, falling back to CPU. Error: {e}")
    classifier = pipeline("image-classification", model="dima806/cat_breed_image_detection", device="cpu")
    print("✅ Model loaded successfully on CPU!")

print("="*50 + "\n")

# --- Inference and Presentation Logic ---
def classify_cat(image):
    if image is None:
        return None, gr.update(visible=False), gr.update(visible=False)
    
    # 1. Run inference
    # Pillow image is passed by Gradio
    results = classifier(image)
    
    # 2. Format outputs for standard labels (Gradio gr.Label)
    label_dict = {res['label']: float(res['score']) for res in results}
    
    # 3. Format dynamic info card for the top prediction
    top_pred = results[0]
    top_breed = top_pred['label']
    top_score = top_pred['score']
    
    # Fetch breed data
    breed_info = BREED_DATABASE.get(top_breed, {
        "origin": "Unknown",
        "temperament": "Information not available",
        "lifespan": "Unknown",
        "fun_fact": "No registered fun facts."
    })
    
    # Custom HTML Card (Tailored Premium Styles)
    html_card = f"""
    <div style="
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(147, 51, 234, 0.08) 100%);
        border: 1.5px solid rgba(79, 70, 229, 0.2);
        border-radius: 16px;
        padding: 24px;
        font-family: 'Outfit', 'Inter', sans-serif;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-size: 12px; font-weight: 700; color: #4f46e5; text-transform: uppercase; tracking-letter: 0.05em;">Identified Breed</span>
                <h3 style="margin: 2px 0 0 0; font-size: 28px; color: #1e1b4b; font-weight: 800; line-height: 1.2;">{top_breed}</h3>
            </div>
            <div style="
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 50px;
                font-weight: 700;
                font-size: 16px;
                box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
            ">
                🔥 Match: {top_score:.1%}
            </div>
        </div>
        
        <div style="
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
            gap: 16px; 
            background: rgba(255, 255, 255, 0.6); 
            padding: 16px; 
            border-radius: 12px;
            border: 1px solid rgba(79, 70, 229, 0.05);
            margin-bottom: 18px;
        ">
            <div>
                <span style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 700;">🌍 Origin</span>
                <div style="font-size: 15px; color: #1f2937; font-weight: 600; margin-top: 4px;">{breed_info['origin']}</div>
            </div>
            <div>
                <span style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 700;">⏳ Lifespan</span>
                <div style="font-size: 15px; color: #1f2937; font-weight: 600; margin-top: 4px;">{breed_info['lifespan']}</div>
            </div>
        </div>
        
        <div style="margin-bottom: 18px;">
            <span style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 700;">🎭 Temperament</span>
            <div style="font-size: 15px; color: #374151; font-weight: 500; margin-top: 4px; line-height: 1.4;">{breed_info['temperament']}</div>
        </div>
        
        <div style="
            background: linear-gradient(90deg, rgba(79, 70, 229, 0.05) 0%, rgba(147, 51, 234, 0.05) 100%);
            border-left: 4px solid #4f46e5;
            padding: 14px;
            border-radius: 0 12px 12px 0;
            font-size: 14px;
            color: #4f46e5;
            line-height: 1.5;
        ">
            <strong style="color: #1e1b4b; display: block; margin-bottom: 4px;">💡 Fun Fact:</strong>
            <span style="color: #4b5563; font-style: italic;">"{breed_info['fun_fact']}"</span>
        </div>
    </div>
    """
    
    return label_dict, gr.update(value=html_card, visible=True), gr.update(visible=True)

# --- Custom Premium Theme and CSS ---
custom_css = """
/* Import sleek modern Google fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Container Styling */
.gradio-container {
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
    background-color: #fafafa !important;
}

/* Header Text styling */
.header-container {
    text-align: center;
    padding: 30px 20px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 20px;
    color: white;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    margin-bottom: 30px;
}
.header-container h1 {
    font-size: 38px !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    letter-spacing: -0.02em !important;
    color: white !important;
}
.header-container p {
    font-size: 16px !important;
    opacity: 0.9;
    margin: 0 !important;
}

/* Cards & Columns styling */
.refined-card {
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    background-color: white !important;
    padding: 10px !important;
}

/* Custom interactive animation for buttons */
button.primary-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
}
button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4) !important;
}
button.primary-btn:active {
    transform: translateY(0) !important;
}

/* Style Example Images to pop out on hover */
.gr-samples-gallery {
    border-radius: 12px !important;
    overflow: hidden;
}
.gr-samples-gallery img {
    transition: transform 0.3s ease !important;
}
.gr-samples-gallery img:hover {
    transform: scale(1.05) !important;
}

/* Output Label list formatting */
.gr-label-output {
    font-family: 'Outfit', sans-serif !important;
    border-radius: 12px !important;
}
"""

# --- Build Gradio App Blocks Layout ---
with gr.Blocks(title="Cat Breed AI Classifier") as demo:
    # 1. Hero Header
    gr.HTML("""
    <div class="header-container">
        <h1>🐱 Cat Breed AI Classifier</h1>
        <p>Upload a photo of your feline friend, and our Vision Transformer (ViT) will identify its breed instantly along with fun facts!</p>
    </div>
    """)
    
    with gr.Row(equal_height=True):
        # Left Column: Input and Examples
        with gr.Column(scale=5, elem_classes="refined-card"):
            gr.Markdown("### 📸 Input Photo")
            image_input = gr.Image(
                type="pil", 
                label="Drop image here or click to upload", 
                sources=["upload", "clipboard", "webcam"],
                show_label=False,
                height=380
            )
            
            submit_btn = gr.Button("🔮 Identify Breed", variant="primary", elem_classes="primary-btn")
            
            # Example triggers
            gr.Markdown("### ✨ Quick Examples")
            gr.Examples(
                examples=[
                    ["sample_cats/siamese.png"],
                    ["sample_cats/persian.png"],
                    ["sample_cats/bengal.png"]
                ],
                inputs=image_input,
                label="Click an example to load & run automatically",
                cache_examples=False
            )
            
        # Right Column: Output Results
        with gr.Column(scale=5, elem_classes="refined-card"):
            gr.Markdown("### 📊 Prediction Results")
            
            # Initially invisible outputs, shown after prediction runs
            info_card = gr.HTML(visible=False)
            
            confidence_chart = gr.Label(
                num_top_classes=5, 
                label="Confidence Scores",
                visible=False,
                elem_classes="gr-label-output"
            )
            
            # Prompt state when no image is uploaded
            no_output_prompt = gr.HTML(
                """
                <div style="
                    text-align: center; 
                    padding: 80px 20px; 
                    color: #9ca3af; 
                    border: 2px dashed #e5e7eb; 
                    border-radius: 12px;
                    font-family: 'Outfit', sans-serif;
                ">
                    <span style="font-size: 48px; display: block; margin-bottom: 12px;">🐈</span>
                    <strong>Waiting for a cat picture...</strong>
                    <p style="margin: 6px 0 0 0; font-size: 14px; color: #a1a1aa;">Upload a photo or select one of the examples on the left.</p>
                </div>
                """,
                visible=True
            )

    # --- Hooking up Action Handlers ---
    
    def on_submit(img):
        if img is None:
            return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        
        # Run classification
        lbl_dict, card_html, gr_lbl_update = classify_cat(img)
        
        # Return results and hide the placeholder prompt
        return lbl_dict, card_html, gr_lbl_update, gr.update(visible=False)
        
    def on_clear():
        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # Click prediction
    submit_btn.click(
        fn=on_submit,
        inputs=image_input,
        outputs=[confidence_chart, info_card, confidence_chart, no_output_prompt],
        api_name="predict"
    )
    
    # Auto-run when example is clicked (Gradio triggers on change)
    image_input.change(
        fn=on_submit,
        inputs=image_input,
        outputs=[confidence_chart, info_card, confidence_chart, no_output_prompt]
    )
    
    # Reset when image is cleared
    image_input.clear(
        fn=on_clear,
        inputs=None,
        outputs=[confidence_chart, info_card, confidence_chart, no_output_prompt]
    )

# --- Startup ---
if __name__ == "__main__":
    # Ensure sample cats folder and files exist
    if not os.path.exists("sample_cats/siamese.png"):
        print("⚠️  Warning: Sample images directory is missing or incomplete.")
        print("Please ensure sample_cats/ contains: siamese.png, persian.png, bengal.png")

    # Launch local web server (automatically opens a browser tab)
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
        css=custom_css
    )
