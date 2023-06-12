# MODIS FSC Retrieval System

## Environment

 - Python 3.9
 - PyTorch 1.13.0
 - Streamlit 1.21.0
 - PyMySQL 1.0.3
 - NumPy 1.21.5
 - Pandas 1.5.3
 - Scikit-Learn 1.1.3
 - Rasterio 1.2.10
 - Rioxarray 0.13.4
 - Plotly 5.13.1

## Database

 - User: root
 - Password: root
 - Database Name: modis
 - Table Name: users
     - username (varchar, key, not null)
     - password (varchar)

## Instructions

 - Run `img2csv.py` can convert images to CSV files.
 - Run `train_nn.py` can train neural network model.
 - Run `train_regressors` can train scikit-learn regressor models.
 - Use `streamlit run app.py` console command can launch Streamlit web server.

## Author's Information

 - Yujie Yi, Lanzhou University of Technology.
 - Links: [Github](https://github.com/AsanoSaki/) | [BLOG](https://asanosaki.github.io/) | [CSDN](https://blog.csdn.net/m0_51755720/)
