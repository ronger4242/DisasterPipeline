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

# Results
Please see the images below:

![overview](https://github.com/ronger4242/DisasterPipeline/blob/master/Images/overview.png)
![classify_info](https://github.com/ronger4242/DisasterPipeline/blob/master/Images/classify_info.png)


# Licensing, Authors, Acknowledgements
This project is licensed under the terms of the MIT license. Credits must be given to Figure Eight project for providing original dataset with Multilingual Disaster Response Messages.
