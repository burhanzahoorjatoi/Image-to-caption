import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VisionAI - Professional Captioner",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    /* Style the caption result box */
    .caption-card {
        padding: 30px;
        border-radius: 15px;
        background-color: #ffffff;
        border-left: 10px solid #FF4B4B;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .result-label {
        color: #888;
        font-size: 0.85em;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }
    .caption-text {
        color: #1f1f1f;
        font-size: 1.5em;
        font-weight: 600;
        line-height: 1.4;
    }
    /* Style the sidebar */
    .css-1d391kg {
        background-color: #111;
    }
    /* Center the button */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #FF4B4B;
        color: white;
        font-size: 1.1em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_vision_model():
    # BLIP is much more professional than older Flickr8K models
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_vision_model()

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚀 VisionAI")
    st.markdown("---")
    section = st.radio("Navigation", ["Home", "Generate Caption", "About Project"])
    st.markdown("---")
    st.info("Using **Salesforce BLIP** Deep Learning Architecture.")
    st.caption("Developed by: Nouman Zahoor Jatoi")

# --- 5. HOME SECTION ---
if section == "Home":
    st.title("🖼️ Image Caption Generator")
    st.subheader("High-Quality Image Descriptions with Deep Learning")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        Welcome! This application interprets visual content and generates professional, 
        human-like captions using state-of-the-art AI.
        
        **Why use this tool?**
        * ✅ **Professional Accuracy:** Uses Beam Search for logical sentences.
        * ✅ **No Repetition:** Advanced penalties prevent word looping.
        * ✅ **Fast Analysis:** Optimized for real-time generation.
        """)
        if st.button("Start Generating →"):
            st.info("Switch to 'Generate Caption' in the sidebar!")
            
    with col2:
        st.image("https://img.freepik.com/free-vector/artificial-intelligence-concept-illustration_114360-7022.jpg", use_column_width=True)

# --- 6. GENERATE CAPTION SECTION ---
elif section == "Generate Caption":
    st.title("🧠 Intelligence Engine")
    st.write("Upload an image and the AI will analyze objects, actions, and the environment.")

    # Main Dashboard Layout
    col_input, col_output = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("📤 Image Upload")
        uploaded_file = st.file_uploader("Drop your image here", type=["jpg", "jpeg", "png"])
        
        with st.expander("⚙️ Fine-Tune AI Settings"):
            # Max tokens controls length
            max_len = st.slider("Max Description Length", 20, 100, 50)
            # Beam search controls quality
            beams = st.slider("Precision (Beam Search)", 1, 10, 5)
            st.caption("Higher precision = Better grammar but slower speed.")

    with col_output:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Preview", use_column_width=True)
            
            # Action Button
            if st.button("✨ Analyze & Generate Caption"):
                with st.spinner("🤖 AI is processing pixels..."):
                    # 1. Prepare image for model
                    inputs = processor(images=image, return_tensors="pt")
                    
                    # 2. ADVANCED GENERATION (Fixes "kurt kurt" and quality issues)
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=max_len,
                        num_beams=beams,           # Beam Search: Explores multiple sentence paths
                        repetition_penalty=1.5,     # Prevents repeating words
                        no_repeat_ngram_size=2,    # Ensures unique word patterns
                        early_stopping=True
                    )
                    
                    # 3. Decode output
                    raw_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    final_caption = raw_text.strip().capitalize()
                    
                    # Grammar cleanup
                    if not final_caption.endswith('.'):
                        final_caption += '.'

                # --- RESULT DISPLAY CARD ---
                st.markdown(f"""
                <div class="caption-card">
                    <div class="result-label">AI Interpretation</div>
                    <div class="caption-text">{final_caption}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Download utility
                st.write("")
                st.download_button(
                    label="💾 Download Description",
                    data=final_caption,
                    file_name="ai_description.txt",
                    mime="text/plain"
                )
        else:
            st.info("Please upload a JPG or PNG image to start the analysis.")

# --- 7. ABOUT SECTION ---
elif section == "About Project":
    st.title("📖 Technical Specifications")
    
    st.markdown("""
    ### How it works
    The application utilizes an **Encoder-Decoder** deep learning architecture.
    """)
    
    
    
    st.markdown("""
    1.  **Encoder (ViT):** The Vision Transformer looks at the image and extracts high-level features.
    2.  **Decoder (Transformer):** A Language Model takes those features and predicts the most likely sequence of words.
    3.  **Beam Search:** Instead of guessing one word at a time, the model evaluates entire sentences to ensure logical flow.
    
    ### Tech Stack
    * **Frontend:** Streamlit (Python)
    * **Model:** Salesforce BLIP (Bootstrapping Language-Image Pre-training)
    * **Engine:** Hugging Face Transformers & PyTorch
    """)