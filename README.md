# Smart-Bin-Classifier

A multimodal AI system that verifies whether the contents of a warehouse bin image match a customer’s order.

- **Computer Vision + NLP + Numeric reasoning**
- **Model:** CLIP-based image–text matcher + MLP for quantity
- **Frontend:** Streamlit web app
- **Backend:** PyTorch, Hugging Face `transformers`

---

## 1. Problem Statement & Objectives

Modern e-commerce fulfillment centers pack multiple items into a single bin. Human or rule-based systems must ensure that:

> **The items and their quantities inside the bin match the customer’s order or invoice.**

This project addresses the following core objective:

> Given a **bin image** and a **customer order** (items + quantities), determine **for each ordered item** whether the bin contains **at least the requested quantity** of that item.

### Key Goals

1. **Understand the dataset & task**  
   - Explore the Amazon Bin-Image Dataset (images + metadata).  
   - Understand the structure of bin contents and item-level metadata.

2. **Build a multimodal model**  
   - Use visual features from bin images.  
   - Use textual descriptions of item names.  
   - Use numeric information (requested quantity).  

3. **Design a practical web UI**  
   - Allow a user to “place an order” from a catalog of items.  
   - Randomly sample a bin image and run the model.  
   - Visually indicate which items/quantities match and which do not.

4. **Document end-to-end ML lifecycle**  
   - Architecture & design decisions  
   - Training, hyperparameters & evaluation  
   - MLOps considerations & future improvements  

---

## 2. Dataset & Preprocessing

> ⚠️ Note: This project uses a **subset** of the original dataset for practicality (e.g., ~25k metadata files and a smaller subset of images for the demo UI).

### 2.1 Source Data

Each bin is represented by:

- **Bin image:** an RGB image showing multiple packed items.
- **Metadata JSON:** structured like:

```json
{
  "BIN_FCSKU_DATA": {
    "B00S81WTMA": {
      "asin": "B00S81WTMA",
      "name": "...",
      "normalizedName": "...",
      "quantity": 2,
      "height": {...},
      "length": {...},
      "width": {...},
      "weight": {...}
    },
    "B014F6ODIY": {
      "asin": "B014F6ODIY",
      "name": "...",
      "normalizedName": "...",
      "quantity": 1,
      ...
    }
  },
  "EXPECTED_QUANTITY": 3
}
```

## 2.2 Metadata Parsing

We build a **flattened item-level table** df_items:

| image_id | asin       | item_name                                      | bin_quantity |
| -------- | ---------- | ---------------------------------------------- | ------------ |
| 12345    | B00S81WTMA | MelodySusie 24W LED Nail Dryer ...             | 2            |
| 12345    | B014F6ODIY | Deconovo Thermal Insulated Grommet Blackout... | 1            |
