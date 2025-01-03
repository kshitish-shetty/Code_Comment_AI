import streamlit as st
from translate import translate
import javalang

# Function to process Java source code
def process_source(code):
    code = code.replace('\n', ' ').strip()
    tokens = list(javalang.tokenizer.tokenize(code))
    tks = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            tks.append('STR_')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            tks.append('NUM_')
        elif tk.__class__.__name__ == 'Boolean':
            tks.append('BOOL_')
        else:
            tks.append(tk.value)
    return " ".join(tks)

# Streamlit App
st.title("Java Code Comment Generator")
st.write("Paste your Java code below and click **Generate Comment** to see the AI-generated comment.")

# Input field for Java code
java_code = st.text_area("Java Code", height=200, placeholder="Type or paste your Java code here...")

# Button to generate comments
if st.button("Generate Comment"):
    if not java_code.strip():
        st.error("Please provide valid Java code!")
    else:
        try:
            processed_code = process_source(java_code)
            generated_comment = translate(processed_code)
            st.subheader("Generated Comment:")
            st.code(generated_comment, language="plaintext")
        except Exception as e:
            st.error(f"An error occurred: {e}")
