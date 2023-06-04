import os

import streamlit as st
import pandas as pd

from uform import get_model
from usearch.index import Index, MetricKind

# Managing data

data_path = os.environ.get('UNSPLASH_SEARCH_PATH')


@st.cache_resource
def get_unum_model():
    return get_model('unum-cloud/uform-vl-english')


@st.cache_resource
def get_unsplash_metadata():
    path = os.path.join(data_path, 'images.csv')
    assert os.path.exists(path), 'Missing metadata file'
    return pd.read_csv(path, dtype=str).fillna('')


@st.cache_resource
def get_unsplash_index():
    index = Index(ndim=256, metric=MetricKind.Cos)
    path = os.path.join(data_path, 'images.usearch')
    assert os.path.exists(path), 'Missing index file'
    index.view(path)
    return index


# GUI setup

st.set_page_config(
    page_title='USearch through Unsplash',
    page_icon='🐍', layout='wide',
    initial_sidebar_state='collapsed',
)

st.title('USearch through Unsplash')

slot_search_bar, _, slot_uform_ai, slot_captions = st.columns((16, 1, 2, 2))

with slot_search_bar:
    query = st.text_input(
        'Search Bar',
        placeholder='Search for Unsplash photos',
        value='', key='query', label_visibility='collapsed')

with slot_uform_ai:
    use_ai = st.checkbox('UForm AI', value=True)

with slot_captions:
    show_captions = st.checkbox('Captions', value=True)

columns = st.sidebar.slider('Grid Columns', min_value=1, max_value=10, value=3)

table = get_unsplash_metadata()
model = get_unum_model()
index = get_unsplash_index()
results = []
max_caption_length = 100
max_results = columns * 20

# Search Content

if not len(query):
    results = table[:max_results]

elif use_ai:
    query_data = model.preprocess_text(query)
    query_embedding = model.encode_text(query_data).detach().numpy()
    matches, _, _ = index.search(query_embedding.flatten(), max_results)
    results = table.iloc[matches]

else:
    results = table[table['photo_description'].str.contains(
        query)][:max_results]

# Visualize Matches

for n_row, row in results.reset_index().iterrows():
    i = n_row % columns
    if i == 0:
        st.write('---')
        cols = st.columns(columns, gap='large')

    with cols[n_row % columns]:
        id = row['photo_id']
        username = row['photographer_username'].strip()

        web_page = row['photo_url']
        preview_path = os.path.join(data_path, 'images', id + '.jpg')
        preview_url = row['photo_image_url']

        if show_captions:
            description = row['photo_description'].strip()
            if len(description) > max_caption_length:
                description = description[:max_caption_length] + '...'
            description = f'{description} \@{username}'
        else:
            description = ''
        st.image(
            preview_path,
            caption=description,
            use_column_width='always')
