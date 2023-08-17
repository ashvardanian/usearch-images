import io
import json
from typing import Dict, List

import numpy as np
import streamlit as st
import PIL as pil
from ucall.client import Client

st.set_page_config(
    page_title="USearch Images",
    page_icon="Ա",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# This following style patch would
style = """
<style>
/* Container */
[role="radiogroup"] {
    display: flex;           /* Use flexbox for layout */
    justify-content: space-between; /* Distribute items evenly across the container */
    width: 100%;             /* Take up the full width of the parent */
    flex-wrap: wrap;         /* Allow items to wrap to the next line if they don't fit */
    gap: 10px;               /* Gap between items */
}
</style>
"""
st.markdown(style, unsafe_allow_html=True)

st.title("USearch Images")

ip_address: str = st.secrets["SERVER_IP"]
ip_address = "0.0.0.0"
ip_address = None


@st.cache_resource
def get_server():
    import server

    return server


@st.cache_resource
def get_examples_by_language() -> Dict[str, List[str]]:
    with open("examples_by_language.json") as f:
        return json.load(f)


@st.cache_resource
def get_examples_vectors() -> Dict[str, np.ndarray]:
    with open("examples_vectors.json") as f:
        return json.load(f)


def unwrap_response(resp):
    if ip_address is None:
        return resp
    else:
        return resp.json


# Starting a new connection every time seems like a better option for now
# @st.cache_resource
# def get_client() -> Client:
#     return Client(uri=ip_address)
client = Client(uri=ip_address) if ip_address is not None else get_server()

image_query = bytes()
text_query = str()
results = list()


examples = get_examples_by_language()
examples_vectors = get_examples_vectors()

# This makes sure the search bar is populated on start,
# even if no text was manually set and no example selected
if "query" not in st.session_state:
    st.session_state["query"] = examples["🇺🇸"][0]

text_query: str = st.text_input(
    "Search Bar",
    placeholder="USearch for Images in the Unsplash dataset",
    value=st.session_state["query"],
    key="text_query",
    label_visibility="collapsed",
)

image_query: io.BytesIO = st.file_uploader("Alternatively, choose an image file")
selected_language = st.radio(
    "Or one of the examples",
    list(examples.keys()),
    horizontal=True,
)


def handle_click(label):
    st.session_state.query = label


example_widths = [len(x) for x in examples[selected_language]]
example_columns = st.columns(example_widths)

for i, example in enumerate(examples[selected_language]):
    with example_columns[i]:
        if st.button(
            example, on_click=handle_click, args=(example,), use_container_width=True
        ):
            text_query = example

columns: int = st.sidebar.slider("Grid Columns", min_value=1, max_value=10, value=8)
max_results: int = st.sidebar.number_input(
    "Max Matches",
    min_value=1,
    max_value=None,
    value=100,
)
dataset_name: str = st.sidebar.selectbox("Dataset", ("unsplash25k",))
size: int = unwrap_response(client.size(dataset_name))


# Search Content

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


st.success(
    f"Displaying {len(results):,} closest results from {size:,} entries!\nUses [UForm](https://github.com/unum-cloud/uform) AI model, [USearch](https://github.com/unum-cloud/usearch) vector search engine, and [UCall](https://github.com/unum-cloud/ucall) for remote procedure calls.",
    icon="✅",
)


# Visualize Matches

for match_idx, match in enumerate(results):
    col_idx = match_idx % columns
    if col_idx == 0:
        st.write("---")
        cols = st.columns(columns, gap="large")

    with cols[col_idx]:
        st.image(match, use_column_width="always")
