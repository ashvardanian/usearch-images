#  Semantic Gallery with StreamLit

```sh
streamlit run streamlit_app.py
```

To serve previews from the local dataset copy, please enable [static file serving](https://docs.streamlit.io/library/advanced-features/static-file-serving):

```sh
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=1
```

## Datasets

All datasets share identical format:

- `images.txt` contains newline-delimited URLs or Base64-encoded data-URIs of images.
- `images.fbin` contains a binary matrix of [UForm][uform] embedding for every image from `images.txt`.
- `images.usearch` contains a binary [USearch][usearch] search index for fast kANN.

Additionally, some image-text paired datasets may provide `texts.txt`, `texts.fbin`, `texts.usearch`, following the same logic.

[uform]: https://github.com/unum-cloud/uform
[usearch]: https://github.com/unum-cloud/usearch

### Unsplash 25K

```sh
sh download.sh unsplash25k
```

Or just directly with `wget`:

```sh
wget -O images.txt https://huggingface.co/datasets/unum-cloud/gallery-unsplash25k/resolve/main/images.txt
wget -O images.fbin https://huggingface.co/datasets/unum-cloud/gallery-unsplash25k/resolve/main/images.fbin
wget -O images.usearch https://huggingface.co/datasets/unum-cloud/gallery-unsplash25k/resolve/main/images.usearch
```

### CC 3M

```sh
sh download.sh cc3m
```

### LAION 400M

```sh
sh download.sh laion400m
```
