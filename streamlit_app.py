import io

import streamlit as st
import PIL as pil
from ucall.client import Client

st.set_page_config(
    page_title='Unum USearch Semantic Gallery',
    page_icon='🐍', layout='wide',
    initial_sidebar_state='collapsed',
)

st.title('Unum USearch Semantic Gallery')

ip_address: str = st.secrets['SERVER_IP']

# Starting a new connection every time seems like a better option for now
# @st.cache_resource
# def get_client() -> Client:
#     return Client(uri=ip_address)
client = Client(uri=ip_address)
size: int = client.size().json

image_query = bytes()
text_query = str()
results = list()


text_query: str = st.text_input(
    'Search Bar',
    placeholder='Search for photos',
    value='Girl in forest', key='text_query',
    label_visibility='collapsed')
if not len(text_query):
    text_query = None

image_query: io.BytesIO = st.file_uploader(
    'Alternatively, choose an image file')

columns: int = st.sidebar.slider(
    'Grid Columns', min_value=1, max_value=10, value=5)
max_results: int = st.sidebar.number_input(
    'Max Matches', min_value=1, max_value=None, value=100)
dataset_name: int = st.sidebar.selectbox(
    'Dataset', (
        'unsplash-25k',
        # Coming soon:
        # 'cc-4m',
        # 'laion-400m',
    ))


# Search Content

with st.spinner(f'We are searching through {size} entries'):

    if image_query:
        image_query = pil.Image.open(image_query).resize((224, 224))
        results = client.find_with_image(
            query=image_query, count=max_results).json
    else:
        results = client.find_with_text(
            query=text_query, count=max_results).json


st.success(
    f'Found {len(results)} matches among {size} entries!',
    icon='✅')


# Visualize Matches

for match_idx, match in enumerate(results):
    col_idx = match_idx % columns
    if col_idx == 0:
        st.write('---')
        cols = st.columns(columns, gap='large')

    with cols[col_idx]:
        st.image(match, use_column_width='always')
