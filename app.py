import streamlit as st
import requests

# Introduction
st.markdown('''## Airline Review Sentiment Classification :airplane:  
Have you ever traveled and had a great experience with the airline? What about a negative experience?  
Test out this airline review classifier! Below, write your airline related review, and this BERT style transformer will classify your review. For example:''')
st.markdown(''' > "I had the worst flight ever. All the flight attendants were rude, and our food was served cold. Do better next time."
            ''', text_alignment="center")
st.markdown("This should come out as :red[negative]. Give your ideas a try!")

MODEL_OPTIONS = {
    "Fine-tuned airline review model": "fine_tuned",
    "Pretrained DistilBERT base model": "pretrained_base",
    "Prebuilt sentiment model (SST-2)": "sentiment_pretrained",
}

selected_model_label = st.selectbox(
    "Choose a model for sentiment evaluation:",
    list(MODEL_OPTIONS.keys()),
    index=0,
)
selected_model_key = MODEL_OPTIONS[selected_model_label]
st.caption(f"Selected model: {selected_model_label}")

# Text box for user input
review_txt = st.text_area(
    "Your Review:",
    "",
    height=2
)

# Button to run review through model & print output
if st.button("Submit"):
    if review_txt.strip() == "":
        st.warning("Please type something first.")
    else:
        api_url = f"http://127.0.0.1:8000/predict/sentiment?model_name={selected_model_key}"
        payload = {"review_text": review_txt}

        try:
            with st.spinner("Communicating to FastAPI backend..."):
                response = requests.post(api_url, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                label = data['prediction']
                score = data['confidence_score']
                model_used = data.get('selected_model', selected_model_label)

                st.success(f"Using model: {model_used}")
                col1, col2 = st.columns(2)
                col1.metric("Sentiment", label)
                col2.metric("Confidence", f"{score}")
            else:
                st.write(f"API Returned error: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

def render_dict(data_dict):
    # Inherits font
    html_content = "<div style='font-family: inherit; line-height: 1.8; font-size: 1rem;'>"
    
    for key, value in data_dict.items():
        # Remove underscores and capitalize
        clean_key = str(key).replace("_", " ").title()
        
        # Set colors
        if isinstance(value, bool):
            color = "#B392F0" 
            val_str = str(value)
        elif isinstance(value, int):
            color = "#E28D5C" 
            val_str = str(value)
        elif isinstance(value, float):
            color = "#B5C442"  
            val_str = str(value)
        elif isinstance(value, str):
            color = "#428ACE" 
            val_str = value
        elif value is None:
            color = "#636e7b"
            val_str = value
        else:
            color = "#B392F0"
            val_str = str(value)
            
        html_content += f"<div style='margin-bottom: 4px;'><span style='color: #F0F0F0; margin-right: 8px;'>{clean_key}:</span> <span style='color: {color};'>{val_str}</span></div>"
    
    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)
with st.expander("What does the classifier architecture look like?"):

    info_url = f"http://127.0.0.1:8000/model/info?model_name={selected_model_key}"
    response = requests.get(info_url)
    if response.status_code == 200:
        model_data = response.json()["data"]
    else:
        st.error(f"Unable to load model info: {response.text}")
        model_data = {}

    if model_data:
        model_data_minimalistic = {
            # Base transformer parameters
            "model_type": model_data.get("model_type"),
            "number_of_layers": model_data.get("n_layers"),
            "number_of_heads": model_data.get("n_heads"),
            "dimensions": model_data.get("dim"),
            "hidden_dimensions": model_data.get("hidden_dim"),
            "activation_function": model_data.get("activation"),
            "dropout": model_data.get("dropout"),
            "attention_dropout": model_data.get("attention_dropout"),
            "vocab_size": model_data.get("vocab_size"),
            "maximum_sequence_length": model_data.get("max_position_embeddings"),
            
            # Classification head & task parameters
            "architecture": model_data.get("architectures")[0] if model_data.get("architectures") else None,
            "problem_type": model_data.get("problem_type"),
            "sequence_classification_dropout": model_data.get("seq_classif_dropout"),
            "padding_token_id": model_data.get("pad_token_id"),
            "id_to_label": model_data.get("id2label")
        }
        
        st.write("""
                 This architecture uses a base transformer model paired with a 
                 task-specific sequence classification head. Input text is tokenized 
                 and passed through a series of transformer layers to build a 
                 contextual understanding of the sequence (aka the text). Once the 
                 model has this representation, it forms a single summary vector. 
                 It then applies a dropout step to prevent overfitting the training data, 
                 and passes that vector through a final layer to output the predicted 
                 class probabilities."
                 """)
        st.write("Below are the core technical specifications and structural parameters of this fine-tuned model:  ")

        render_dict(model_data_minimalistic)
    else:
        st.warning("Model information is unavailable at this time.")
    
    # The longer version of model_data is formatted as such:

        # {
        # "vocab_size":30522
        # "max_position_embeddings":512
        # "sinusoidal_pos_embds":false
        # "n_layers":6
        # "n_heads":12
        # "dim":768
        # "hidden_dim":3072
        # "dropout":0.1
        # "attention_dropout":0.1
        # "activation":"gelu"
        # "initializer_range":0.02
        # "qa_dropout":0.1
        # "seq_classif_dropout":0.2
        # "pad_token_id":0
        # "eos_token_id":NULL
        # "bos_token_id":NULL
        # "tie_word_embeddings":true
        # "return_dict":true
        # "output_hidden_states":false
        # "dtype":"float32"
        # "chunk_size_feed_forward":0
        # "is_encoder_decoder":false
        # "architectures":[
        # 0:"DistilBertForSequenceClassification"
        # ]
        # "id2label":{
        # "0":"LABEL_0"
        # "1":"LABEL_1"
        # "2":"LABEL_2"
        # }
        # "label2id":{
        # "LABEL_0":0
        # "LABEL_1":1
        # "LABEL_2":2
        # }
        # "problem_type":"single_label_classification"
        # "_name_or_path":"lindybujak/airline-review-modified"
        # "transformers_version":"5.0.0"
        # "model_type":"distilbert"
        # "tie_weights_":true
        # "output_attentions":false
        # }