# Yelp Stars Prediction


<p align="center">

![imagen_readme.jpeg](./images/readme_image.png)

</p>





## Project



This is the final project for the subject Machine Learning of CUNEF Master´s in Data Science. The objective of this project is to study the [Yelp dataset](https://www.yelp.com/dataset/download), find business cases and to create predictive models.


## Business Case


The business objective is through the information registered by the businesses on the platform and their characteristics, to be able to predict whether the average score they will achieve from the users will be high (>=4 stars) or low (<4 stars).


To achieve this, a preprocessing of the data has been carried out, where JSON files have been treated and exported as a parquet.  Then, an exploratory analysis of the data, the creation of pipelines to treat the selected variables according to the type of data, and testing of different models.

Finally, we proceed to calculate the local and global explainability to obtain the importance of the variables, also we created a graph with a business case applicable to the set used in our model.

## How to run the Project?


To run the project, you should install the environment writing in the shell:

pip3 install -r requirements.txt

Then, you should download the Yelp dataset, extract and move it to the data/raw folder.

## What did we use?


- Python 3.9.13

- Visual Studio Code

- Jupyter Notebook

- Networkx

## Index



0. Data Preprocessing

1. EDA
2. Feature Engineering
3. Models

    - Base Model (Dummy Model)
    - Logistic Regression Lasso
    - Random Forest
    - Light GBM
    - Support Vector Machine
    - XGBoost


4. Model Selection
5. Interpretability
6. Graph 

## Content of the repository



- `data`:

	- raw: Documents downloaded from the source of the dataset.

	- processed: Data dictionay processed and data processed. 
	
    - maps: Map to load the stars numbers by state
    
    - graphs: Folder where the graph will be exported.


- `images`: Pictures used in the differents notebooks.



- `html`: Notebooks exported as html files.



- `notebooks`: Notebooks of the project and functions .py files.



- `models`: Pickles of the different models. 

- `env`: Requirements of the environment.



## Authors



Victor Viloria Vázquez 

- Email: victor.viloria@cunef.edu

- LinkedIn: https://www.linkedin.com/in/vicviloria/





Antonio Nogués Podadera:

- Email: antonionpodadera@gmail.com

- LinkedIn: https://www.linkedin.com/in/antonio-nogu%C3%A9s-podadera/



## Project Link: 

https://github.com/ComputingVictor/Yelp_Stars
