r"""
Created on 22/6/2021 1:59 PM
@author: jiahuei
"""
import urllib.request
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image

METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]


def dict_filter(dict_obj, key_list):
    return {k: v for k, v in dict_obj.items() if k in key_list}


@st.cache(show_spinner=True)
def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        img = Image.open(BytesIO(response.read()))
    return img
