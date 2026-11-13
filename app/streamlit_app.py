import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so `streamlit run app/streamlit_app.py` finds our packages
from inference.generate import load_checkpoint, generate_text
from inference.attention_viz import plot_attention

st.set_page_config(page_title="nanoGPT", layout="wide")
st.title("nanoGPT explorer")

ckpt_dir = "checkpoints"
ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt")) if os.path.isdir(ckpt_dir) else []
if not ckpts:
    st.warning("no checkpoints found, train one first: python -m training.train --config configs/small.yaml")
    st.stop()

# model selection: every checkpoint carries its own config and tokenizer, so a char model, a bpe model and a rope model all load the same way
choice = st.sidebar.selectbox("model", ckpts)


@st.cache_resource  # loading is slow, so we only do it once per checkpoint
def get_model(name):
    return load_checkpoint(os.path.join(ckpt_dir, name), 'cpu')


model, tokenizer, cfg = get_model(choice)
m = model.metrics
st.sidebar.markdown(f"**tokenizer:** {cfg.tokenizer} (vocab {tokenizer.vocab_size})  \n"
                    f"**size:** {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} dims, context {cfg.chunk_size}  \n"
                    f"**position:** {cfg.pos_encoding}, **activation:** {cfg.activation}")
if m:
    st.sidebar.markdown(f"**params:** {m['n_params']:,}  \n**val loss:** {m['val_loss']:.3f}  \n**val bits/char:** {m['val_bpc']:.3f}")  # bits/char is the number that is comparable across tokenizers

tab1, tab2 = st.tabs(["generate", "attention"])
with tab1:
    prompt = st.text_area("prompt", "ROMEO:")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("max new tokens", 50, 1000, 300)  # tokens, not characters: a bpe model writes ~2 chars per token
    temperature = c2.slider("temperature", 0.0, 2.0, 1.0, 0.05, help="0 = always the most likely token (greedy)")
    top_k = c3.number_input("top-k (0 = off)", min_value=0, max_value=tokenizer.vocab_size, value=0)
    if st.button("generate"):
        try:
            st.code(generate_text(model, tokenizer, prompt, n, 'cpu', temperature=temperature, top_k=int(top_k) or None))
        except KeyError:
            st.error("the prompt has a character this model's vocabulary doesn't contain")  # only the char model can hit this
with tab2:
    sentence = st.text_input("sentence", "To be, or not to be, that is the question.")
    layer = st.slider("layer", 0, cfg.n_layer - 1, 0)
    try:
        st.pyplot(plot_attention(model, tokenizer, sentence, layer))
    except KeyError:
        st.error("the sentence has a character this model's vocabulary doesn't contain")
