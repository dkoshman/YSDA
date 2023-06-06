from constants import RESOURCES
from data_preprocessing import RandomizeImageTransform
from utils import beam_search_decode

import streamlit as st
import PIL
import torch
import torchvision.transforms as T

MODEL_PATH = RESOURCES + "/model_2tcuvfsj.pt"

transformer = torch.load(MODEL_PATH)
image_transform = T.Compose(
    (
        T.ToTensor(),
        RandomizeImageTransform(
            width=transformer.hparams["image_width"],
            height=transformer.hparams["image_height"],
            random_magnitude=0,
        ),
    )
)

st.title("Image to TeX")

st.image("resources/frontend/fraction_derivative.png", width=500)
st.image("resources/frontend/positional_encoding.png")
st.image("resources/frontend/taylor_sequence_expanded.png")
# st.image("resources/frontend/taylor_sequence.png")
# st.image("resources/frontend/maclaurin_series.png")
# st.image("resources/frontend/gauss_distribution.png")

image_file = st.file_uploader(
    "Upload an image with equation", type=([".png", ".jpg", ".jpeg"])
)

if image_file is not None:
    image = PIL.Image.open(image_file)
    image = image.convert("RGB")
    texs = beam_search_decode(transformer, image, image_transform=image_transform)
    # streamlit latex doesn't support boldmath
    tex = texs[0].replace("\\boldmath", "")
    st.latex(tex)
    st.markdown(tex)
