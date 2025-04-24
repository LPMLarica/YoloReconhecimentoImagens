import streamlit as st 
from transformers import YolosForObjectDetection
from transformers import AutoImageProcessor

from PIL import Image, ImageDraw
import torch 

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")


st.title("Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Performing Object Detection...")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist() 
        box = [round(i, 2) for i in box]  
        
        draw.rectangle(box, outline="red")  
        draw.text(box[:2], f"{model.config.id2label[label.item()]} {score.item():.2f}", fill="red")

        
    st.image(image, caption="Detected Objects", use_column_width=True)
    
    st.write("Detected Objects:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]  
        st.write(f"Detected **{model.config.id2label[label.item()]}** "
                f"with confidence **{round(score.item(), 3)}** "
                f"at location **{box}**")
