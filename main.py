import os

import streamlit as st
import pandas as pd

from uform import get_model
from usearch.index import Index, MetricKind

# Managing data

script_path: str = os.path.dirname(os.path.abspath(__file__))

data_path: str = os.environ.get(
    'UNSPLASH_SEARCH_PATH', os.path.join(script_path, 'dataset'))

view_local_images: bool = True if os.environ.get(
    'STREAMLIT_SERVER_ENABLE_STATIC_SERVING') else False


@st.cache_resource
def get_uform_model():
    return get_model('unum-cloud/uform-vl-english')


@st.cache_resource
def get_unsplash_metadata():
    path = os.path.join(data_path, 'images.csv')
    assert os.path.exists(path), 'Missing metadata file'
    df = pd.read_csv(path, dtype=str).fillna('')
    df['photo_submitted_at'] = pd.to_datetime(
        df['photo_submitted_at'], format='mixed')
    return df


@st.cache_resource
def get_unsplash_usearch_index():
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

slot_search_bar, _, slot_layout, slot_uform_ai, slot_captions = st.columns(
    (16, 1, 2, 2, 2))

with slot_search_bar:
    query: str = st.text_input(
        'Search Bar',
        placeholder='Search for Unsplash photos',
        value='', key='query',
        label_visibility='collapsed')

with slot_uform_ai:
    use_ai: bool = st.checkbox('UForm AI', value=True)

with slot_captions:
    show_captions: bool = st.checkbox('Captions', value=True)

with slot_layout:
    layout: str = st.radio(
        'Layout',
        ('List', 'Grid'),
        horizontal=True,
        label_visibility='collapsed')

columns: int = st.sidebar.slider(
    'Grid Columns', min_value=1, max_value=10, value=3)
max_results: int = st.sidebar.number_input(
    'Max Matches', min_value=1, max_value=None, value=100)

model = get_uform_model()
table = get_unsplash_metadata()
index = get_unsplash_usearch_index()

results = []
max_caption_length = 100

# Search Content

if not len(query):
    results = table[:max_results]

else:
    with st.spinner(f'We are searching through {len(table)} entries'):

        if use_ai:
            query_data = model.preprocess_text(query)
            query_embedding = model.encode_text(query_data).detach().numpy()
            matches, _, _ = index.search(
                query_embedding.flatten(), max_results)
            results = table.iloc[matches]

        else:
            results = table[table['photo_description'].str.contains(
                query)][:max_results]

    st.success(f'Found {len(results)} results!', icon='✅')


# Join metadata with images
results = results.copy().reset_index()
results['photo_image_path'] = [
    os.path.join(data_path, 'images', id + '.jpg')
    for id in results['photo_id']]

# Visualize Matches
if layout == 'List':

    columns = [
        'photo_id',
        'photo_url',
        'photo_image_url',
        'photo_image_path',
        'photo_description',
        'ai_description',
        'photographer_username',
        'photo_submitted_at',
        'stats_views',
        'stats_downloads',
    ]
    visible_results = results[columns]

    st.data_editor(
        visible_results,
        column_config={
            'photo_id': st.column_config.TextColumn('ID'),
            'photo_url': st.column_config.LinkColumn('Page'),
            'photo_image_url': st.column_config.ImageColumn(
                'Remote',
                width='medium'),
            'photo_image_path': st.column_config.ImageColumn(
                'Local',
                width='medium'),
            'photo_submitted_at': st.column_config.DatetimeColumn(
                'Time',
                format='DD.MM.YYYY',
            ),
            'photo_description': st.column_config.TextColumn('Human Text'),
            'ai_description': st.column_config.TextColumn('AI Text'),
            'photographer_username': st.column_config.TextColumn('Author'),
            'stats_views': st.column_config.NumberColumn('Views'),
            'stats_downloads': st.column_config.NumberColumn('Downloads'),
        },
        use_container_width=True,
        disabled=True,
        hide_index=False,
    )
else:
    for n_row, row in results.reset_index().iterrows():
        i = n_row % columns
        if i == 0:
            st.write('---')
            cols = st.columns(columns, gap='large')

        with cols[n_row % columns]:
            id = row['photo_id']
            username = row['photographer_username'].strip()
            preview_path = row['photo_image_path']

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
