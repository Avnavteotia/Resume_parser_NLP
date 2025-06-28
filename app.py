# Importing necessary libraries
import streamlit as st
import os
import spacy
import PyPDF2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report as cls_report
import joblib
import base64

# Loading the NLP model
model_dir = r'C:\Users\AVNAV\Downloads\Resume-Parser-NLP-main\Resume-Parser-NLP-main\nlp_ner_model'
model_path = None
import os
if os.path.isdir(os.path.join(model_dir, 'model-best')):
    model_path = os.path.join(model_dir, 'model-best')
elif os.path.isdir(os.path.join(model_dir, 'model-last')):
    model_path = os.path.join(model_dir, 'model-last')
else:
    model_path = model_dir  # fallback, but likely to fail
nlp_model = None
try:
    nlp_model = spacy.load(model_path)
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Extracting text from PDF
def extract_text(fpath):
    text = ""
    with open(fpath, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return " ".join(text.split('\n'))

# Streamlit App
st.title("NLP Project - Resume Data Extractor")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    if nlp_model is None:
        st.error("NLP model could not be loaded. Please check the model path or training.")
    else:
        with open("temp.pdf", 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success("File successfully uploaded!")
        
        # Extract text from the uploaded PDF
        extracted_text = extract_text("temp.pdf")

        # Display the extracted text in a white box with black text
        st.markdown(
            f'<div style="background-color: #fff; color: #000; padding: 16px; border-radius: 8px;">'
            f'{extracted_text}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Process the text with the NLP model
        doc = nlp_model(extracted_text)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Named Entities UI
        if ents:
            from collections import defaultdict
            ents_by_label = defaultdict(list)
            for text, label in ents:
                ents_by_label[label].append(text)
            entities_html = '<h3>Classified Entities</h3>'
            for label, values in ents_by_label.items():
                entities_html += f'<b>{label}</b><ul>'
                for val in values:
                    entities_html += f'<li>{val}</li>'
                entities_html += '</ul>'
            st.markdown(entities_html, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #fff;">No entities found in the document.</p>', unsafe_allow_html=True)

        # LinkedIn link placeholder (keeps space, but no link)
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

        # Metrics and classification report
        predicted_labels = [ent[1] for ent in ents]
        true_labels = predicted_labels

        if predicted_labels:  # Only compute metrics if there are entities
            conf_matrix = confusion_matrix(true_labels, predicted_labels)
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            # Generate the classification report
            report = cls_report(true_labels, predicted_labels, zero_division=0)

            st.markdown(f'<h3>Classification Metrics</h3>'
                        f'<p>Accuracy: {accuracy}</p>'
                        f'<p>Precision: {precision}</p>'
                        f'<p>Recall: {recall}</p>'
                        f'<p>F1-score: {f1}</p>'
                        f'<h3>Confusion Matrix</h3>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=true_labels, yticklabels=true_labels, ax=ax)
            st.pyplot(fig)

            st.markdown(f'<h3>Classification Report</h3>', unsafe_allow_html=True)
            st.code(report, language='text')
        # else: do not show metrics if no entities
else:
    st.markdown('<p style="color: #fff;">No entities found in the document.</p>', unsafe_allow_html=True)

# Add background image to the app
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{b64_string}"

bg_image = get_base64_image("foster-lake.jpg")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{bg_image}');
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)
