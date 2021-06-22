r"""
Created on 22/6/2021 4:18 PM
@author: jiahuei

streamlit run explore_json.py
"""
import json
import streamlit as st


# https://discuss.streamlit.io/t/issue-with-caching-data-uploaded-via-the-file-uploader/3791/3
@st.cache(allow_output_mutation=True)
def load_json(file):
    return json.load(file)


def main():
    st.sidebar.title("JSON Explorer")
    st.sidebar.markdown("---")

    # COCO JSON for image URLs
    upload_help = "Provide JSON file"
    uploaded_file = st.sidebar.file_uploader(upload_help)
    if uploaded_file is None:
        st.info(f"{upload_help}, by uploading it in the sidebar")
        return

    data = load_json(uploaded_file)
    st.header(f"Filename: {uploaded_file.name}")
    st.markdown(f"---")
    top1, top2 = st.beta_columns([5, 6])
    with top1:
        st.info('You can provide an expression for filtering the loaded JSON "`data`" object.')
    with top2:
        st.warning('Note: The expression is evaluated using `eval()`. See https://stackoverflow.com/a/34385055')
    expression = st.text_input(
        label='Expression examples: data["x"], data[0]',
        value="data"
    )
    st.markdown(f"---")
    try:
        data = eval(expression, {"data": data, "__builtins__": {}})
    except Exception:
        st.markdown(f"Invalid expression. The full JSON object is:")
        st.json(data)
        raise
    else:
        st.markdown(f"Length: {len(data)}")
        st.markdown(f"Data:")
        st.json(data)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
