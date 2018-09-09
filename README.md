## Data Science Project Guidelines, Version 2.0.41

##### Data science projects require structure.  In essence, you are solving data problems by applying the scientific method.  The following guidelines have been refined with over 2,000 data science students, and is a continuous iteration for building capstone projects for your data science or data engineering portfolio.  If you would like to contribute to the material, you are welcome to share across your considerations, and offer these guidelines with attribution to your colleagues as well.

### Part One

<details>
  <summary><strong>Proposal Requirements :notebook:</strong></summary>

> It is recommended to create several ideas (3 to 5) each on a PowerPoint slide that includes the following components below, to facilitate the scoping of your data science project.
> After you have identified the project you will work on, then create a markdown (.md) file on Github using the same structure more in-depth, such as an Abstract/Thesis for your project (ideally 300-600 words).  [This is a markdown reference guide.](https://youtu.be/V0fZkWDkPLA)

These criteria are strongly encouraged in scoping out your project: 

> 1. Project Name: What is the Title for your project 
> 2. Problem Statement: What are you trying to solve? (High-level overarching problem to be explored)
> 2. Data science Hypothesis(es)/solutions considering: One or multiple angles to consider solving this problem with data science.  This could be as many as 10+ questions for what you are considering to solve through analysis, visualizations, and machine learning 
> 3. Data sets to be used: Share the links and files, .csvs, .js, .xlsx, .txt, website URLs.  If you are usng web scraping or APIs, note the appropriate links and strategies here as well. Also describe how large are your data files and if you are concerned on any shortcomings of your data.
> 4. Data dictionaries to describe the data types you're using: write these out in markdown as tables  
> 5. Potential business cases relative to project: how would this help businesses out to make money or save money or improve accuracy or make better products
> 6. Potential stakeholders who would find this interesting: who would be your ideal customer or client for this?
> 7. Potential places to share your results post project (client, world, website, blog)

> This information can be presented in a PowerPoint presentation.  Consider applying the **SMART** Framework as well as Data Science Methodology to structure for projects with Specific, Measurable, Attainable, Reproducible, and Time-bound goals.  Describe how you will **Frame, Prepare, Analyze, Interpret, and Communicate** your data science challenge.
-------------------------------------------------------------------------------------------------------------------

**Deliverables:**
> 1. To be pushed to github
> 2. To be submitted as a markdown readme.md file in your project-final repository
</details>


### Part Two
<details>
  <summary><strong>EDA Guidelines :bar_chart:</strong></summary>
  
> 0. Creating a capstone project can result in stress levels on your machine that cause slow processing power.  In order to accelerate your prototyping during the development phase, you can consider a cloud solution offering such as [Google Colab](https://colab.research.google.com/), [Microsoft Notebooks](https://notebooks.azure.com/) and [IBM Notebooks](https://dataplatform.cloud.ibm.com/docs/content/analyze-data/notebooks-parent.html).  If you are looking for more advanced infrastructure, consider providers such as [Amazon Web Services](https://aws.amazon.com/), [Microsoft Azure](http://azure.microsoft.com/), [Google Cloud Platform](https://cloud.google.com/gcp), and [IBM Watson Data Studio](https://www.ibm.com/cloud/watson-studio).  If you are looking for instant container solutions for data science projects, consider [Crestle](https://www.crestle.com/) and [Paperspace](https://www.paperspace.com/)
> 1. To start, please be sure to create Notebooks that you code your data analysis in.  These notebooks should be through the Jupyter framework, which support .ipynb (iPython Notebooks) and .md (Markdown) files, as well as interactivity between both, which can be displayed effectively through a Github environment or [Binder](https://mybinder.org/). If you would like, you can do additional editing through [VSCode](https://vscodecandothat.com/) and even set it as your [default editor](https://stackoverflow.com/questions/30024353/how-to-use-visual-studio-code-as-default-editor-for-git).
> 2. Import your data or data files and to save as dataframes, and convert [XML to DataFrames](http://www.austintaylor.io/lxml/python/pandas/xml/dataframe/2016/07/08/convert-xml-to-pandas-dataframe/) when needed.  And [unzip files](https://chrisjean.com/unzip-multiple-files-from-linux-command-line/) easily.
> 3. Examine your data, columns and rows and rename and adjust indexing and encoding as appropriate 
> 4. Clean null and blank values, and consider to drop rows, as well as to manipulate data and adjust data types as appropriate, including [dates](https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html) and [time](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html), or setting appropriate indices. Adjusting specific values and replacing strings and characters for the data wrangling process.  
> 5. Explore analysis with graphing and visualizations with matplotlib and seaborn and alternative visualization packages ([Plot.ly and Dash](https://plot.ly/products/dash/), [Bokeh](https://bokeh.pydata.org/en/latest/), [Altair](https://altair-viz.github.io/), [Vincent](https://vincent.readthedocs.io/en/latest/), [pygal](http://pygal.org/en/stable/))
> 6. Perform additional analysis by creating new columns for calculations, including aggregator functions, counts and groupbys.
> 7. Encode categorical variables with a variety of techniques through logical conditions, mapping, applying, where clauses, dummy variables, and one hot encoding.  When displaying results, consider to [format](https://pyformat.info/) them as well.
> 8. Re-run calculations, including crosstabs or pivots, and new graphs to see results 
> 9. Create correlation matrices, [pairplots](https://seaborn.pydata.org/generated/seaborn.pairplot.html), scatterplot matrices, and [heatmaps](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to determine which attributes should be features for your models and which attributes should not.  Design your visualizations with themes such as [pallettes](https://seaborn.pydata.org/tutorial/color_palettes.html). 
> 10. Identify the response variables(s) that you would want to predict/classify/interpret with data science 
> 11. Perform additional feature engineering as necessary, including Min/Max, Normalizaton, Scaling, and additional Pipeline changes that may be beneficial or helpful when you run machine learning 
> 12. Merge or concatenate datasets with [Pandas merging](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html), or SQL methods (I.e., [Learning SQL](https://www.quora.com/What-some-of-the-websites-where-I-can-practice-Advance-SQL), [SQL Joins](http://sqlhints.com/tag/cross-join/), [Joins #2](https://stackoverflow.com/questions/38549/what-is-the-difference-between-inner-join-and-outer-join), [Joins #3](https://stackoverflow.com/questions/17759687/cross-join-vs-inner-join-in-sql-server-2008), [SQL Tutorial](https://community.modeanalytics.com/sql/tutorial/introduction-to-sql/), and [Saving Queries](https://stackoverflow.com/questions/31769736/saving-sql-queries-as-sql-text-file) if you have not already, based on common keys or unique items for more in-depth analysis 
> 13. Add commenting and markdown throughout the jupyter notebook to explain the interpretation of your results or to comment on code that may not be human readable, and help you recall for you what you are referencing.  
> 14. To create a markdown .md milestone report that shows and explains the results of what you have accomplished to date in this part of your course project. Consider also creating a .pdf or .pptx to display initial results, aha moments, or findings that would be novel or fascinating for your final presentations. 
</details>

### Part Three
<details>
  <summary><strong>Machine Learning Guidelines :computer:</strong></summary>
  
> 0. Create a brand new Jupyter notebook, where you run the latest DataFrame or .csv files(s) that you have previously saved from your exploratory data analysis notebook. 
> 1. After you have completed the exploratory data analysis section of your project, start revisiting your hypothesis(es) on ideas that you would like to either predict (regression) or classify (classifier).  > 2. Have you identified a specific column or multiple columns that could be treated as response or target variables to predict/classify?
> 3. If not, consider performing additional exploratory analysis that helps you pinpoint a potential working hypothesis to test results against. You could consider clustering as an addition to exploratory data analysis as a preparation for machine learning.
> 4. Consider for your machine learning what parts of your feature engineering have been completed, or need to additionally be completed through [Pre-processing](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) or [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) operations such as Normalize, Scaler, Min/Max, etc. 
> 5. As a result of correlation matrices, heatmaps, and visualizations, consider which features may be relevant to support the model that you are building. 
> 6. Consider what machine learning models through [SkLearn](http://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf) and their [Github Repo](https://github.com/scikit-learn/scikit-learn) or [StatsModels](https://www.statsmodels.org/stable/index.html) could be effective for your newly discovered hypothesis (linear regressions (I.e., [Lowess Regression](http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html), logistic regression, KNearest Neighbors, [Clustering](http://scikit-learn.org/stable/modules/clustering.html), Decision Trees including [Bagging Regressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html) or the [Bagging Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html), Random Forest including [Tuning RF](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74), Naive Bayes, Natural Language Processing ([Word2Vec](https://github.com/davidyakobovitch/word2vec-translation), [Spacy](https://github.com/davidyakobovitch/spaCy-tutorial), and [Topic Modeling](https://github.com/davidyakobovitch/topic-modeling)) Time Series Analysis, [Neural Networks](http://scikit-learn.org/stable/modules/neural_networks_supervised.html), Support Vector Machines and [Model Resistance](http://scikit-learn.org/stable/modules/model_persistence.html), [Stochastic Gradient Descent](http://www.scikit-learn.org/stable/modules/sgd.html), dimensionality reduction with PCA as well as Ensembles such as [GB Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)).  Once you have determined models to consider, be sure to import their packages into Python.
> 7. Consider what tuning parameters you may want to optimize for your model, including regularization (Lasso, ridge, ElasticNet), and additional parameters relevant to each model.
> 8.  Be sure to include a train_test_split, and then consider a KFolds or Cross Validations to offer stratified results that limit the interpretation of outliers for your dataset.
> 9. If you still have many outliers, consider how to remove them or optimize for them with categories.  How could you adjust your categories, or thresholds to improve performance for what you are testing for your hypothesis? Depending on how your model error performs, you may want to consider to change or adjust other features in your model.  You may want to consider to add or remove features, and measure the feature importance when running models. 
> 10.  Consider a [Grid Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [Random Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to better optimize your models. 
> 11.  Share metrics on each model that is run, such as error and accuracy, confusion matrices, and the [ROC/AUC](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) scores.  Other models have additional metrics, that you can consider to share.  You can set up metrics and running models in defined functions for further automation of your project. 
> 12. Compare your metrics against the base case or null case for accuracy, which ideally is compared to your majority class, or a median/mean representation for your target/response variable.  How well does your model perform?
> 13. Provide markdown explaining the interpretation relevant to your business case after running models.  Also, share comments to explain what you are doing, for your interpretation and then reproducibility of your code. 
> 14.  If you are running Time Series Analysis, you will want to consider additional model capabilities such as rolling and moving averages with the dateTime package and pandas.
> 15.  If you are working on Natural Language processing, you will want to consider python packages such as [Spacy](https://github.com/davidyakobovitch/spaCy-tutorial), [topic modeling](https://github.com/davidyakobovitch/topic-modeling),  NLTK, TextBlob, and [word2vec](https://github.com/davidyakobovitch/word2vec-translation).
> 16. If you are scraping additional data, consider python packages such as Selenium and BeautifulSoup4.
> 17.  For your project, your presentation will showcase the best 3-5 models.  However, it is fine if you have inefficient models that do not perform well, for practice, so keep these in your main modeling Jupyter notebook as a reference. 
</details>

### Part Four
<details>
  <summary><strong>Presentation Guidelines :information_desk_person:</strong></summary>

#### Content Guidelines
> 1. Cover page aligned to your project theme or organizational template 
> 2. Table of contents page that discusses the slides covered in your report 
> 3. At a minimum, presentation should include the following slides below:
> 4. Problem/Hypothesis(es) page that you explored/offering a solution too
> 5. Data dictionary page, describing your data
> 6. Exploratory Data Analysis page, describing data wrangling, feature engineering, and cleaning performed on data
> 7. Data Visualization pages (no more than 2 visualizations shown on each page for readibility) describing fascinating, insightful visualizations that indicate trends, novel interpretations, or offer clarity and context for your business case.  Either in this section, or the next one to define your baseline/null accuracy for the majority class of hypothesis you are testing to improve results in your scoring.
> 8. Machine learning page(s) that describe the 3 to 5 best performing models for your project including their metrics and inteprretability to business case. 
> 9. Summary table that compares your 3 to 5 best machine learning models side-by-side and which model overall performed the best to solve your hypothesis and direct results for your stakeholder or client
> 10. Concluding remarks on your project and next steps/recommendations slide.
> 11. Appendix slide that includes Bibliography for research, references, works cited, dataset links, and Github links  

#### Design Requirements
> 0. Reorganize all files on Github Project Final as a folder for data, a folder for images saved as assets, readme(s).md, and .ipynb notebooks starting with naming convention 00_datawrangling, 01_eda, 02_visualizations, 03_machinelearning, or similar, etc.
> 1. To be saved as both .pptx or .key files to Github, as well as the final .pdf presentation file, including a .md markdown Abstract Milestone Report.
> 2. Code to not be shown in presentation except where necessary to convey an explanation 
> 3. No more than 3 fonts to be used throughout the entire presentation.
> 4. Presentation will be between 8 to 20 slides.
> 5. Presentation format should be self-explained, such that a stakeholder can read the report without you physically being present to explain it. 
> 6. Presentation should include an Appendix slide that documents resources including dataset links used for your project. 

#### Product Delivery
> 1. Presentation to be delivered in a 7 to 8 minute format. 
> 2. Presentation to be delivered for a non-technical stakeholder/client. 
> 3. All explanations to be related to business case, intepretability, and impact for the business. 
> 4. Software to delivery to include a screen-share software, including an on-screen annotation delivery (I.e., Open Board, Zoom)
> 5. Presentation leaves room for Questions & Answers and Feedback session for at most 2 to 3 additional minutes.
</details>

## Licenses
License

[![CC-4.0-by-nc-nd](https://licensebuttons.net/l/by-nc-nd/3.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

To the extent possible under law, [David Yakobovitch](http://davidyakobovitch.com/) has licensed this work under Creative Commons, 4.0-NC-ND.  This [license](https://creativecommons.org/licenses/by-nc-nd/4.0/) is the most restrictive of Creative Commons six main licenses, only allowing others to download your works and share them with others as long as they credit the author, but they can’t change them in any way or use them commercially.
