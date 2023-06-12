from numpy import tile
import streamlit as st

class MultiPage:
    def __init__(self) -> None:
        self.pages = []

    def add_page(self, title, func):
        self.pages.append(
            {
                'title': title,
                'function': func
            }
        )

    def run(self):
        st.sidebar.subheader('APP Navigation')
        page = st.sidebar.selectbox(
            'Select Page',
            self.pages,
            format_func=lambda page: page['title']
        )
        st.sidebar.write('---')
        page['function']()
