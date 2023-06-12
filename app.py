import streamlit as st
import pymysql
import time
from multipage import MultiPage
from other_pages import information_page, machine_learning_page, deep_learning_page
from PIL import Image

st.set_page_config(page_title='MODIS FSC Retrieval APP', page_icon=':snowflake:', layout='wide')

text_column, image_column = st.columns(2)
lut_logo = Image.open('images/lut_logo.png')
with text_column:
    st.markdown("<h1 style='text-align: left; color: black;'>MODIS FSC Retrieval Application</h1>", unsafe_allow_html=True)
with image_column:
    st.image(lut_logo, width=350)
# st.title('MODIS FSC Retrieval Application')

app = MultiPage()

app.add_page('Information', information_page.app)
app.add_page('Machine Learning', machine_learning_page.app)
app.add_page('Deep Learning', deep_learning_page.app)

con = pymysql.connect(host="localhost", user="root", password="root", database="modis", charset="utf8")
c = con.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS users(username VARCHAR, password VARCHAR)')

def add_userdata(username, password):
    if c.execute('SELECT username FROM users WHERE username = %s', (username)):
        st.warning("The user name already exists. Please replace it with a new one.")
    else:
        c.execute('INSERT INTO users(username, password) VALUES(%s, %s)', (username, password))
        con.commit()
        st.success("Register successfully!")
        st.info("Please select the \"Login\" option on the left to log in.")

def login_user(username,password):
    if c.execute('SELECT username FROM users WHERE username = %s', (username)):
        c.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        data = c.fetchall()
        return data
    else:
        st.warning("The user name does not exist. Please click the \"Register\" button to register.")

def login():
    menu = ['Login', 'Register']

    if 'status' not in st.session_state:
        st.session_state.status = False

    if st.session_state.status:
        st.sidebar.success('User has logged in')
        app.run()
        if st.sidebar.button('Logout'):
            st.session_state.status = False
            # st._rerun()
            st.experimental_rerun()
        return

    choice = st.sidebar.selectbox('Choose a servey', menu)

    if choice == 'Login':
        st.sidebar.subheader('Login')
        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.button('Login'):
            if login_user(username, password):
                st.session_state.status = True
                if st.session_state.status:
                    st.sidebar.success(f'Welcome: {username}!')
                    time.sleep(2)
                    # st._rerun()
                    st.experimental_rerun()
                    # app.run()
            else:
                st.sidebar.warning('Username or password is wrong, please check and try again!')

    elif choice == 'Register':
        st.sidebar.subheader('Register')
        new_user = st.sidebar.text_input('Username')
        new_password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.button('Register'):
            # create_usertable()
            add_userdata(new_user, new_password)

if __name__ == '__main__':
    # app.run()
    login()