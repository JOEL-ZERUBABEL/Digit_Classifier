import numpy as np
import streamlit as st
from mlp_model import MLP
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit page setup
st.set_page_config(page_title='Digit Classifier', layout='centered')
st.title('🧠 Handwritten Digit Classifier')
st.write('Upload an image of a handwritten *digit (0–9)* to get its prediction.')

class Page:

    @staticmethod
    def model(img):
        """Load model, preprocess image, predict digit & probabilities."""
        mlp_digit_classifier = MLP()
        mlp_digit_classifier.load('mlp_model_weights.npz') 
        img = img.convert('L').resize((8, 8))
        img_array = 16 * (1 - np.array(img) / 255.0)   # inverted + rescaled
        img_array = img_array.flatten().reshape(1, -1)

        st.caption(f"Image Stats → min={img_array.min():.3f}, max={img_array.max():.3f}, mean={img_array.mean():.3f}")

        probs = mlp_digit_classifier.forward(img_array)
        pred = np.argmax(probs, axis=1)

        return pred[0], probs[0]

    @staticmethod
    def file_upload():
        """Handles image upload and triggers prediction."""
        uploaded_file = st.file_uploader("📤 Upload your handwritten digit image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='🖼 Uploaded Digit', width=250)

            pred, probs = Page.model(img)
            st.success(f"✅ Predicted Digit: *{pred}*")
            Page.plot(probs)

    @staticmethod
    def plot(probs):
        """Plots probability bar chart for each class (0–9)."""
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(range(10), probs, color='teal', edgecolor='black')
        ax.set_xticks(range(10))
        ax.set_xlabel('Digit Classes (0–9)')
        ax.set_ylabel('Prediction Confidence')
        ax.set_title('Prediction Probabilities')
        st.pyplot(fig)


if __name__ == '__main__':
    Page.file_upload()