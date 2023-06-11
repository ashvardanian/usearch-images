#  Semantic Gallery with StreamLit

```sh
streamlit run main.py
```

To serve previews from the local dataset copy, please enable [static file serving](https://docs.streamlit.io/library/advanced-features/static-file-serving):

```sh
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=1
```

## Unsplash Dataset

To visualize the unsplash dataset, download it from HuggingFace and link it under the `dataset/` path.

```sh
wget -O dataset/.csv https://huggingface.co/datasets/unum-cloud/unsplash-search/resolve/main/images.csv
wget -O dataset/.fbin https://huggingface.co/datasets/unum-cloud/unsplash-search/resolve/main/images.fbin
wget -O dataset/.usearch https://huggingface.co/datasets/unum-cloud/unsplash-search/resolve/main/images.usearch
wget -O dataset/.zip https://huggingface.co/datasets/unum-cloud/unsplash-search/resolve/main/images.zip
```
