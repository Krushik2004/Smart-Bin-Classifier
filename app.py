import os
import random
import json

import streamlit as st
import pandas as pd
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel
from rapidfuzz import process

from model import CLIPQuantityMatcher  # import your model class
import base64


st.markdown("""
    <style>
    .stButton > button {
        background-color: #2196F3;   
    }
    .stButton > button:hover {
        background-color: #0b7dda;
    }
    </style>
""", unsafe_allow_html=True)



# ---------- CONFIG ----------

ITEMS_FILE = "unique_item_names.json"  # list of ~38k unique item names
IMAGES_DIR = "bin-images"                      # folder with ~100 bin images
MODEL_WEIGHTS = "head_weights.pt"           # your saved head weights

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


# ---------- HELPERS ----------

@st.cache_data
def load_item_names():
    with open(ITEMS_FILE, "r") as f:
        items = json.load(f)
    # Ensure all are strings and sorted
    items = sorted(set(str(x) for x in items))
    return items


@st.cache_resource
def load_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP processor + model
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_model.to(device)
    clip_model.eval()

    # Init your quantity-attention model
    model = CLIPQuantityMatcher(
        # clip_model_name=CLIP_MODEL_NAME,
        # freeze_clip=True,  # CLIP is frozen
    )

    # Load head weights (only custom layers)
    state = torch.load(MODEL_WEIGHTS, map_location=device)
    model.quantity_mlp.load_state_dict(state["quantity_mlp"])
    # model.cls_token.data = state["cls_token"]
    # model.ln1.load_state_dict(state["ln1"])
    # model.attn.load_state_dict(state["attention"])
    model.classifier.load_state_dict(state["classifier"])

    model.to(device)
    model.eval()

    # Attach the CLIP backbone
    # model.clip = clip_model

    return model, processor, device


def predict_match(model, processor, device, image_path, item_name, quantity):
    """
    Runs the model on (image, item_name, quantity) and returns:
    - 1 if present (prob >= 0.5)
    - 0 otherwise
    """
    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        text=[item_name],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    pixel_values = encoding["pixel_values"].to(device)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    qty_tensor = torch.tensor([float(quantity)], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(pixel_values, input_ids, attention_mask, qty_tensor)
        prob = torch.sigmoid(logits).item()

    return 1 if prob >= 0.5 else 0, prob


def get_random_image_path():
    candidates = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not candidates:
        return None
    filename = random.choice(candidates)
    return os.path.join(IMAGES_DIR, filename)


# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="Bin Order Validator", layout="wide")
st.title("📦 Bin Image & Order Validator")

# Load resources
item_names = load_item_names()
model, processor, device = load_model_and_processor()

# Initialize session state
if "order_items" not in st.session_state:
    # list of dicts: { "item": str, "quantity": int }
    st.session_state.order_items = []

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = None


st.markdown("### 🛒 Create an Order")

# Layout: [Quantity dropdown] [Search bar + suggestions]
qty_col, search_col = st.columns([1, 3])

with qty_col:
    quantity = st.selectbox(
        "Quantity",
        options=list(range(1, 21)),
        index=0,
        key="qty_select",
    )

with search_col:
    selected_item = st.selectbox(
        "Select Item:",
        options=item_names,     # full list of 38k names
        index=0,
        key="item_dropdown",
    )


# Buttons row
btn_add_col, btn_done_col, btn_clear_col = st.columns([1, 1, 1])



def add_current_item():
    item = st.session_state.item_dropdown
    qty = st.session_state.qty_select

    # Enforce max 10 unique items
    existing_items = [x["item"] for x in st.session_state.order_items]
    if item in existing_items:
        st.warning("This item is already in your order.")
        return

    if len(existing_items) >= 10:
        st.warning("You can only add 10 items.")
        return

    st.session_state.order_items.append({"item": item, "quantity": qty})


with btn_add_col:
    if st.button("➕ Add item"):
        add_current_item()

with btn_done_col:
    done_clicked = st.button("Done")

with btn_clear_col:
    clear_clicked = st.button("Clear")


# ---------- ORDER TABLE (current order) ----------

if st.session_state.order_items:
    st.markdown("### 📝 Current Order")
    order_df = pd.DataFrame(st.session_state.order_items)
    order_df = order_df.rename(columns={"item": "Item", "quantity": "Quantity"})
    st.table(order_df)
else:
    st.info("No items in the order yet. Add items using the search bar above.")


# ---------- WHEN DONE IS CLICKED ----------

if done_clicked and st.session_state.order_items:
    # Optionally add the last selected item if not already added
    # (only if user forgot to click "Add another item" before "Done")
    if st.session_state.selected_suggestion is not None:
        # Try to add it if unique and not exceeding the limits
        add_current_item()

    st.markdown("---")
    st.markdown("## 🔍 Model Validation")

    # Re-show final order
    # final_order = pd.DataFrame(st.session_state.order_items)
    # final_order = final_order.rename(columns={"item": "Item", "quantity": "Quantity"})
    # st.subheader("Order Summary")
    # st.table(final_order)

    # Pick a random bin image from 100 images
    image_path = get_random_image_path()
    if image_path is None:
        st.error("No images found in the images directory.")
    else:
        st.subheader("Selected Bin Image")
        st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/jpg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" 
             alt="Bin Image" 
             style="width: 224px; height: 224px;" />
    </div>
    """,
    unsafe_allow_html=True
)

        # Run model for each item
        statuses = []
        probs = []
        for row in st.session_state.order_items:
            item_name = row["item"]
            qty = row["quantity"]
            label, _ = predict_match(model, processor, device, image_path, item_name, qty)
            statuses.append("✅" if label == 1 else "❌")
            # probs.append(round(prob, 3))

        result_df = pd.DataFrame({
            "Item": [x["item"] for x in st.session_state.order_items],
            "Quantity": [x["quantity"] for x in st.session_state.order_items],
            "Match": statuses,
            # "Confidence": probs,
        })

        st.subheader("Model Output")
        st.table(result_df)

# ---------- CLEAR BUTTON FUNCTIONALITY ----------
if clear_clicked:
    st.session_state.order_items = []
    st.session_state.search_query = ""
    st.session_state.selected_suggestion = None

    # Remove image and output states if you store them
    if "selected_image" in st.session_state:
        del st.session_state["selected_image"]

    if "model_results" in st.session_state:
        del st.session_state["model_results"]

    st.rerun()
