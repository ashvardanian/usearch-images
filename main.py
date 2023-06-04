import os

import streamlit as st
import pandas as pd

from uform import get_model
from usearch.index import Index

# Page setup
st.set_page_config(
    page_title='USearch through Unsplash',
    page_icon='🐍', layout='wide')
st.title('USearch through Unsplash')

data_path = '/Users/av/github/unsplash-search/'
if 'table' not in st.session_state:
    st.session_state['table'] = pd.read_csv(
        os.path.join(data_path, 'images.csv'),
        dtype=str).fillna('')

if 'model' not in st.session_state:
    st.session_state['model'] = get_model('unum-cloud/uform-vl-english')

if 'index' not in st.session_state:
    index = Index(ndim=256, metric='cos')
    index.view(os.path.join(data_path, 'images.usearch'))
    st.session_state['index'] = index


query = st.text_input('Search for Unsplash photos', value='', key='query')
columns = st.slider('Grid Columns', min_value=1, max_value=10, value=3)
use_ai = st.checkbox('AI')
show_captions = st.checkbox('Captions')


table = st.session_state['table']
model = st.session_state['model']
index = st.session_state['index']
results = []
max_caption_length = 100
max_results = columns * 20

if not len(query):
    results = table[:max_results]
elif use_ai:
    query_data = model.preprocess_text(query)
    query_embedding = model.encode_text(query_data).detach().numpy()
    matches, _, _ = index.search(query_embedding.flatten(), max_results)
    results = table.iloc[matches]
    print('matches are', matches, results)
else:
    results = table[table['photo_description'].str.contains(
        query)][:max_results]

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
