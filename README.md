# Disaster Response Pipeline Udacity

# Description and Goals:

This project provides a webinterface for analyze disaster data from Figure Eight. A model for an API that classifies disaster messages has been build.
A data set contains real messages that were sent during disaster events. Therefore a machine learning pipeline have been created to categorize these events so that you can send the messages to an appropriate disaster relief agency.
The project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

# Includes:

    * **process_data.py**: This code will perform a data preprocessing due to the CRISP-DM process and creates a SQL database
    * **train_classifier.py**: This code trains a KNN model
    * **ETL** Pipeline Preparation.ipynb: This is the baseline for the process_data.py development procces
    * **ML** Pipeline Preparation.ipynb: This is the baseline for the train_classifier.py. development procces
    * **data**: This folders contains the CSV data for messages and categories.
    * **app**: cointains the run.py for the web app.


### Instructions for execution:

1. Run the following commands in the **project's root directory** to set up your database and model.

    - To run **ETL pipeline** that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run **ML pipeline** that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the **app's directory** to run your **web app**.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to see the web app.

# File Description:

The **notebooks folder** contains two jupyter notebooks that help you understand how the pipeline scripts are built step by step:

- **ETL Pipeline Preparation**: Loads the datasets, merges them, cleans the data and stores them in a SQLite database.
- **ML Pipeline Preparation**: Loads the dataset from SQLite database, splits data into train and test set, builds a text preprocessing and ML pipeline, trains and tunes models using GridSearch (SVM, Random Forest), outputs reults on the test set and exports the final model as a pickle file.

**Python scripts**:

- `data/process_data.py` - ETL pipeline 
- `models/train_classifier.py` - ML Pipeline
- `app/run.py` - Flask Web App

**Datasets**:

- **messages.csv**: Contains the id, message and genre, i.e. the method (direct, social, ...) the message was sent.
- **categories.csv**: Contains the id and the categories (related, offer, medical assistance..) the message belonges to.

# Licensing
This project is a part of the **Data Science Nanodegree** on [Udacity](https://www.udacity.com).
The data have been provided by [Figure Eight](https://appen.com) and provided by [Udacity](https://www.udacity.com).

