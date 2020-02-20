# DisasterPipeline
This is the second project for Udacity Nano degree in Data Science.
# Installation
Python 3.x is required to run this Project. The following libraries are used:
* Pandas
* Numpy
* Matplotlib
* Plotly
* Sklearn
* NLTK
* Pickle
* Re

# Project Motivation
Following natural disasters, there are overwhelmingly amount of messages. This project tends to build a data pipeline to help the disaster organization to filter and organize the messages, and thus faciliate the resources allocation process. 

# File Description
This project contains the following folders and files:
* app
   * templates
      * master.html (main page of web app)
      * go.html (classification result page of web app)
   * run.py (Flask file that runs app)
* data
   * disaster_categories.csv (data to process)
   * disaster_messages.csv (data to process)
   * process_data.py
   * DisasterResponse.db

* models
   * train_classifier.py
   * classifier.pkl (saved model)

* README.md (project instructions)
# How to run:

1. To run ETL pipeline that cleans data and stores in database:<br />
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db<br />

2. To run ML pipeline that trains classifier and saves the model:<br />
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl<br />

3. Run the following command in the app's directory to run the web app:<br />
python run.py<br />

4. To get WORKSPACEID and WORKSPACEDOMAIN:<br />
run env | grep WORK<br />
Then replace the http://WORKSPACEID-3001.WORKSPACEDOMAIN with respective terms.<br />
http://view6914b2f4-3001.udacity-student-workspaces.com<br />

# Results
Please see the images below:<br />

![overview](https://github.com/ronger4242/DisasterPipeline/blob/master/Images/overview.png)<br />



# Licensing, Authors, Acknowledgements
This project is licensed under the terms of the MIT license. Credits must be given to Figure Eight project for providing original dataset with Multilingual Disaster Response Messages.
