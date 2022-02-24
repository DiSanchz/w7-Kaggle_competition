# Kaggle competition - Airbnb prices
## Repository made as part of the Ironhack Data Analytics bootcamp program

The present repository contains the code written and employed for a machine learning competition whose goal is to develop a machine learning model capable of predicting Airbnb prices for the city of Amsterdam. The score is assessed by the RMSE and Python is the programming language tu be employed.


### > The Process

A pipeline has been devolped in the different notebooks that can be found in this repository. Firstly the raw dataset which serves as starting point for the competition is loaded by the 'Processing' notebook, whose function is (as its name states) to process, clean and shape the data that is later to be used for prediction. In the 'Processing' notebook, most of the process is standardized, however it is designed in such way that allows to easily remove/include variables in order to be able to test different data configuration in an agile way within the jupyer framework.

After the data set is shaped, it is exported to .csv and then reimported by one or both of the notebooks devoted to train the machine learning models, the 'Modeling' and the 'H2O' notebooks. 'Modeling' is centered on linear-regression alike models and random forests powered by SKlearn, while 'H2O' employs the H2O python library in order to authomatize the generation and selection of a richer range of models.

Finally the 'Testing' notebook is the one in charge to generate the predictions and export them to a format compatible with the one required within the rules of the competition.

More detailed information on the process can be found in each indevidual notebook's comments.


### > Selected Technology Stack:

- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciKit Learn](https://scikit-learn.org/stable/)
- [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html)


### > The Repository

Besides the before-mentioned jupyter notebooks, you can also find in this repo a src folder containing a .py file that includes some ad-hoc-developed functions using during the data cleaning and shaping stage as well as the raw train and test datasets.

