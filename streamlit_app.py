import io
import os
import json
from typing import Dict, List

import numpy as np
import streamlit as st
import PIL as pil
from ucall.client import Client

# Set Streamlit page configuration
st.set_page_config(
    page_title="USearch Images",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Style patch for radio buttons
st.markdown(
    """
    <style>
    [role="radiogroup"] {
        display: flex;
        justify-content: space-between;
        width: 100%;
        flex-wrap: wrap;
        gap: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set up Google Analytics
google_analytics = (
    st.secrets.get("GOOGLE_ANALYTICS", None)
    if os.path.exists(".streamlit/secrets.toml")
    else None
)
if not google_analytics:
    google_analytics = "G-71Y7G3QKXQ"

st.markdown(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={ga}"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', '{ga}');
    </script>
    """.format(
        ga=google_analytics
    ),
    unsafe_allow_html=True,
)

st.title("USearch Images")

# Server IP address
ip_address = (
    st.secrets.get("SERVER_IP", None)
    if os.path.exists(".streamlit/secrets.toml")
    else None
)


@st.cache_resource
def get_server():
    """Load server module locally, if no remote IP is set."""
    import server

    return server


@st.cache_resource
def get_examples_by_language() -> Dict[str, List[str]]:
    """Load language examples from JSON."""
    with open("assets/examples_by_language.json") as f:
        return json.load(f)


@st.cache_resource
def get_examples_vectors() -> Dict[str, np.ndarray]:
    """Load example vectors from JSON."""
    with open("assets/examples_vectors.json") as f:
        return json.load(f)


def unwrap_response(resp):
    """Extract response data based on the presence of IP address."""
    return resp if ip_address is None else resp.json


# Initialize primary variables
client = Client(uri=ip_address) if ip_address else get_server()
examples_by_language = get_examples_by_language()
examples_vectors = get_examples_vectors()

# Primary UI elements for search input
st.session_state.setdefault("query", examples_by_language["🇺🇸"][0])
text_query = st.text_input(
    "Search Bar",
    placeholder="USearch for Images",
    value=st.session_state["query"],
    key="text_query",
    label_visibility="collapsed",
)
image_query = st.file_uploader("Alternatively, choose an image file")
selected_language = st.radio(
    "Or one of the examples",
    list(examples_by_language.keys()),
    horizontal=True,
)


def handle_click(label):
    """Update session state with selected example."""
    st.session_state.query = label


examples_widths = [len(x) for x in examples_by_language[selected_language]]
examples_columns = st.columns(examples_widths)

for i, example in enumerate(examples_by_language[selected_language]):
    with examples_columns[i]:
        if st.button(
            example, on_click=handle_click, args=(example,), use_container_width=True
        ):
            text_query = example

# UI elements for search configuration are snucked into the side bar
columns: int = st.sidebar.slider("Grid Columns", min_value=1, max_value=10, value=8)
max_results = st.sidebar.number_input("Max Matches", min_value=1, value=100)
dataset_name: str = st.sidebar.selectbox(
    "Dataset",
    (
        "unsplash-25k",
        "cc-3m",
    ),
)
size = unwrap_response(client.size(dataset_name))

# Perform search, showing a spinning wheel in the meantime
with st.spinner(f"We are searching through {size:,} entries"):
    if image_query:
        image_query = pil.Image.open(image_query).resize((224, 224))
        results = unwrap_response(
            client.find_with_image(
                dataset=dataset_name,
                query=image_query,
                count=max_results,
            )
        )
    # Avoid AI inference if we can :)
    elif text_query in examples_vectors.keys():
        results = unwrap_response(
            client.find_vector(
                dataset=dataset_name,
                vector=np.array(examples_vectors[text_query]),
                count=max_results,
            )
        )
    else:
        results = unwrap_response(
            client.find_with_text(
                dataset=dataset_name,
                query=text_query,
                count=max_results,
            )
        )

# Display Results
st.success(
    f"Displaying {len(results):,} closest results from {size:,} entries.",
    icon="✅",
)

st.markdown(
    """
    This page serves as a proof-of-concept for constructing a multi-lingual, multi-modal Semantic Search system with just 50 lines of Python code. 
    The content is encoded using the {uform} multi-modal transformer model, which produces embeddings in a shared vector space for text queries and visual content.
    These embeddings are then indexed and queried via the {usearch} vector search engine.
    For client-server interaction, the stack utilizes {ucall} for remote procedure calls.
    Each component in this architecture is optimized for high throughput, making it scalable for large datasets.
    After the initial retrieval, results can be further refined using larger neural network models before being presented to the end-user.
    """.format(
        uform="[![UForm stars](https://img.shields.io/github/stars/unum-cloud/uform?style=social&label=UForm)](https://github.com/unum-cloud/uform)",
        usearch="[![USearch stars](https://img.shields.io/github/stars/unum-cloud/usearch?style=social&label=USearch)](https://github.com/unum-cloud/usearch)",
        ucall="[![UCall stars](https://img.shields.io/github/stars/unum-cloud/ucall?style=social&label=UCall)](https://github.com/unum-cloud/ucall)",
    )
)

max_rows = 4

for match_idx, match in enumerate(results[: columns * max_rows]):
    col_idx = match_idx % columns
    if col_idx == 0:
        cols = st.columns(columns, gap="small")
    with cols[col_idx]:
        st.image(match, use_column_width="always")
