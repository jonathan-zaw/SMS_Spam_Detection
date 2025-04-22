# conda activate base
# pip install -U streamlit
# pip install -U plotly
# Run the app using: streamlit run app.py

import streamlit as st
import pandas as pd
import pickle
import time
import re

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Error: 'model.pkl' not found. Make sure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading the model: {e}")
    st.stop()

# Feature extraction function (must match training)
def extract_features(df):
    df['has_link'] = df['message'].str.contains(r'http|www|\.com|\.ru|\[link\]', regex=True).astype(int)
    df['has_money'] = df['message'].str.contains(r'\$\d+', regex=True).astype(int)
    df['has_urgent_words'] = df['message'].str.contains(r'urgent|required|immediately|clearance|customs|win', case=False, regex=True).astype(int)
    return df

# App title
st.title('📩 Spam Message Classifier')

# User input
message = st.text_input('Enter a message', value='', placeholder='Type your message here...')

# Prediction button
if st.button('Predict'):
    if message.strip():  # Ensure it's not empty or just whitespace
        with st.spinner('🔍 Analyzing message...'):
            time.sleep(1)

            # Prepare data for prediction
            test_df = pd.DataFrame({'message': [message]})
            test_df = extract_features(test_df)

            # Predict
            prediction = model.predict(test_df[['message', 'has_link', 'has_money', 'has_urgent_words']])

        st.toast('✅ Prediction completed!')

        # Show result
        if prediction[0] == 'spam':
            st.warning('🚫 This message is SPAM')
        else:
            st.success('✅ This message is Legit (HAM)')
    else:
        st.warning('⚠️ Please enter a message to predict.')

# Footer
st.markdown("---")
st.caption("Model is trained to classify messages as Spam or Legitimate (HAM).")