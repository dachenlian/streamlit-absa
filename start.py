import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Movie Review Explorer",
    page_icon="🎬",
)

if "url" not in st.session_state:
    st.session_state["url"] = ""

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""


def callback():
    """This is needed because Streamlit resets all session state variables that come from widgets after every action."""
    sess_url = st.session_state["_url"]
    sess_api_key = st.session_state["_api_key"]

    if not sess_api_key:
        st.error("Please enter an API key.")
        return

    if not sess_url:
        st.error("Please enter a URL.")
        return

    st.session_state["url"] = sess_url
    st.session_state["api_key"] = sess_api_key


st.title("Sentiment Analysis Explorer")

with st.form("best_form"):
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        help="Starts with sk-",
        key="_api_key",
        type="password",
    )
    url = st.text_input(
        "Enter a URL (IMDb, Letterboxd, PTT)",
        help="The link should go directly to a post or reviews.",
        key="_url",
        # on_change=callback,
    )

    submitted = st.form_submit_button(
        "Submit",
        on_click=callback,
    )

if st.session_state["url"] and st.session_state["api_key"]:
    switch_page("explore")