#  Semantic Gallery with StreamLit

```sh
streamlit run streamlit_app.py
```

To serve previews from the local dataset copy, please enable [static file serving](https://docs.streamlit.io/library/advanced-features/static-file-serving):

```sh
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=1
```

## Unsplash Dataset

To visualize the unsplash dataset, download it from HuggingFace and link it under the `dataset/` path.

```sh
wget -O images.txt https://huggingface.co/datasets/unum-cloud/gallery-unsplash25k/resolve/main/images.txt
wget -O images.fbin https://huggingface.co/datasets/unum-cloud/gallery-unsplash25k/resolve/main/images.fbin
wget -O images.usearch https://huggingface.co/datasets/unum-cloud/gallery-unsplash25k/resolve/main/images.usearch
```
