import io
import os
import base64
import mimetypes

import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
import sklearn as skl
import PIL as pil

from uform import get_model
from usearch.index import Index, MetricKind
from ucall.client import Client

st.write('Hello!')

ip_address = st.secrets['SERVER_IP']
st.write(ip_address)
client = Client(uri=ip_address, port=8545)

response = client.find(request='Ash')

st.write(response)


# Managing data

script_path: str = os.path.dirname(os.path.abspath(__file__))

data_path: str = os.environ.get(
    'UNSPLASH_SEARCH_PATH', os.path.join(script_path, 'dataset'))

view_local_images: bool = True if os.environ.get(
    'STREAMLIT_SERVER_ENABLE_STATIC_SERVING') else False


image_query = bytes()
text_query = str()
results = list()
max_caption_length: int = 100


class FileNotFoundError(Exception):
    pass


def img_to_data(path):
    """Convert a file (specified by a path) into a data URI."""
    if not os.path.exists(path):
        raise FileNotFoundError
    mime, _ = mimetypes.guess_type(path)
    with open(path, 'rb') as fp:
        data = fp.read()
        data64 = base64.b64encode(data).decode('utf-8')
        return f'data:{mime}/jpg;base64,{data64}'


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


def draw_gui():

    st.set_page_config(
        page_title='USearch through Unsplash',
        page_icon='🐍', layout='wide',
        initial_sidebar_state='collapsed',
    )

    st.title('USearch through Unsplash')

    text_query: str = st.text_input(
        'Search Bar',
        placeholder='Search for Unsplash photos',
        value='', key='text_query',
        label_visibility='collapsed')
    if not len(text_query):
        text_query = None

    image_query: io.BytesIO = st.file_uploader(
        'Alternatively, choose an image file')

    layout: str = st.radio(
        'Layout',
        ('Grid', 'List', 'Semantics'),
        horizontal=True,
        label_visibility='collapsed')

    columns: int = st.sidebar.slider(
        'Grid Columns', min_value=1, max_value=10, value=5)
    show_captions: bool = st.sidebar.checkbox(
        'Show Captions in Grid', value=True)
    max_results: int = st.sidebar.number_input(
        'Max Matches', min_value=1, max_value=None, value=100)

    model = get_uform_model()
    table = get_unsplash_metadata()
    index = get_unsplash_usearch_index()

    # Search Content

    if not text_query and not image_query:
        results = table[:max_results]

    else:
        with st.spinner(f'We are searching through {len(table)} entries'):

            if image_query:
                image_query = pil.Image.open(image_query)
                query_data = model.preprocess_image(image_query)
                query_embedding = model.encode_image(
                    query_data).detach().numpy()
            else:
                query_data = model.preprocess_text(text_query)
                query_embedding = model.encode_text(
                    query_data).detach().numpy()

            # We don't need the text-based search, if we have AI :)
            # results = table[table['photo_description'].str.contains(
            #     text_query)][:max_results]
            matches, _, _ = index.search(
                query_embedding.flatten(),
                max_results,
                exact=True,
            )
            results = table.iloc[matches]

        st.success(
            f'Found {len(results)} matches among {len(table)} entries!', icon='✅')

    # Join metadata with images

    results = results.copy().reset_index()
    results['photo_image_base64'] = [
        img_to_data(os.path.join(data_path, 'images', id + '.jpg'))
        for id in results['photo_id']]

    # Visualize Matches

    if layout == 'List':

        columns = [
            'photo_image_base64',
            'photo_description',
            'ai_description',
            'photographer_username',
            'photo_submitted_at',
            'stats_views',
            'stats_downloads',
            'photo_id',
            'photo_url',
            'photo_image_url',
        ]
        visible_results = results[columns]

        st.dataframe(
            visible_results,
            column_config={
                'photo_id': st.column_config.TextColumn('ID'),
                'photo_url': st.column_config.LinkColumn('Page'),
                'photo_image_url': st.column_config.LinkColumn('Remote'),
                'photo_image_base64': st.column_config.ImageColumn(
                    'Local',
                    width='large'),
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
            hide_index=False,
            height=1000,
        )

    elif layout == 'Semantics':

        vectors = np.vstack([index[id] for id in results['photo_id']])
        tsne = skl.manifold.TSNE(
            n_components=2, learning_rate='auto',
            init='random', perplexity=3).fit_transform(vectors)

        results['x'] = tsne[:, 0]
        results['y'] = tsne[:, 1]

        altair_chart = alt.Chart(results).mark_circle(size=200).encode(
            x='x',
            y='y',
            tooltip=['photo_image_base64'],
        )
        st.altair_chart(altair_chart, use_container_width=True,
                        theme='streamlit')

    elif layout == 'Grid':

        for n_row, row in results.reset_index().iterrows():
            i = n_row % columns
            if i == 0:
                st.write('---')
                cols = st.columns(columns, gap='large')

            with cols[n_row % columns]:
                id = row['photo_id']
                username = row['photographer_username'].strip()
                preview_path = row['photo_image_base64']

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


if False:
    draw_gui()
