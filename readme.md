<font size="6em"><strong>Data Science Standards</strong></font>
<br>
<strong>What are the Data Science Standards?</strong>
The Data Science Standards are a proven process at both the university and the bootcamp level for students to create production grade machine learning for their portfolio, to excel in the job interview.  This process has been stress-tested with over 5,000 students and offers you the following:

<ul>
<li> A Framework that leads to confidence with client success and career interviews</li>
<li> A portfolio to share as Proof of Concepts to clients and for career opportunities</li>
<li> A Standard for mental model and business framework to solving production grade machine learning </li>
<li> An organized, and centralized repository for state of the art resources for production grade data science</li>
<li> Available for any technology stack</li>
</ul>

Foundational Literature:
1. [How to ask Data Science questions](https://towardsdatascience.com/how-to-ask-questions-data-science-can-solve-e073d6a06236)
2. [Asking insightful questions](https://amplitude.com/blog/2015/07/01/question-the-data-how-to-ask-the-right-questions-to-get-actionable-insights)
3. [Data Questionnaire](https://www.fast.ai/2020/01/07/data-questionnaire/)

## Data Science Project Deliverables:
1. Part 1: Project Proposal Criteria - Prepare an Abstract as both a Document and a PowerPoint (Start with 3 to 6 project ideas)
2. Part 2: Perform Exploratory Data Analysis, Visualizations, and Feature Engineering
3. Part 3: Perform Machine Learning, Performance Metrics, and Deployment for your project 
4. Part 4: Present your project as a Presentation to your business stakeholders
5. Part 5: Submit your project for your Advisors and business stakeholders

### Part 1: Project Proposal Guidelines:
> _Please prepare your project proposal as a sharable document, and a PowerPoint presentation_
1. Project Title
- What is your Project Theme (I.e., Industry Vertical/Machine Learning Topic)?
- What is your Abstract? Write a 1-paragraph Executive Summary of your Solution.
2. Problem Statement & Business Case
- What is the technical problem you are solving?
- What is the applied business case for this problem? 
  - Business perspective (I.e., Likelihood, sentiment, demand, price, market strategy, groups, automation)
3. Data Science Workflow 
- What Null/Alternative Hypothesis are you testing against?
  - Does X Predict Y? (I.e., Distinct groups, key components, outliers)
- What is the response column/predictor that is important for you to measure?
- What assumptions are important for you to assess and to benchmark?
- What solutions would you like to deliver against?
- How will you measure your benchmarks and their performance drift over time (I.e., Automate jobs/predictive monitoring)?
- What alternative questions would you like to explore and provide solutions?
- What analytics and insights would you like to discover from your data? - What types of graphics or machine learnings would you like to discover?
- What is the business case for your project? 
- How will your solution help generate revenue, reduce costs, or impact another Key Performance Indicator or Objective Key Result?
- Who will be impacted (Executive Stakeholders/Sponsors) by your solution? Who is your ideal client/customer?
4. Data Collection
- What raw datasets/APIs will you extract for machine learning?
- What are the data schemas for your current datasets (I.e., SQL, CSVs, Parquet, Avro, Snowflake/Star)
- What are the dimensions and sizing (I.e., MB/GB/TB/PB) of your current datasets?
- Is the data from open-source, paid crowdsourcing, internal?
- What is the structures, file types, and quality of the data?
- How will you collect, store, and process the data (I.e., locally, databases, cloud)?
- Of your known data, what is the current data dictionaries that exist, or that you can further describe? (You can create these data dictionaries in a spreadsheet, markdown table, or listed)
5. Data Processing, Preparation, & Feature Engineering
- What techniques will you use to improve your data quality?
- How will you handle missing data and outliers?
- What calculations/formulas would you like to create, that may not yet exist?
6. Machine Learning: Model Selection
- Which model architecture(s) will you use to solve your problem?
- How will you validate the model performance?
7. Model Persistence: Deployment, Training, & Data Pipelines
- How would your results operate LIVE in a production environment? (I.e., Web App, Architecture flow, DAG diagram, end-to-end workflow)
- What technology stack, what integrations, and which Engineers would you cooperate?
- Where will you share your results internally or externally to stakeholders through Marketing, Implementation and Deployments?
- How will you validate your machine learnings with a timeline from development to production?  How will you generate more data to train?

<hr size="3">

### Part 2: Exploratory Data Analysis Guidelines:
The Exploratory Data Analysis is a significant progression from Defining a Data Science Problem to determine the specific characteristics needed to solve the problem.  From Data Wrangling, Data Munging, Pre-processing, Pipelines, Data Visualization, and Data Analytics, all these areas are essential for effective Exploratory Data Analysis.
 
> **1. Compute and Storage Considerations:** 
_Projects that scale require more compute, faster computer, and more storage. In the market, many solutions from many providers exist.  If you need Cloud Compute and Storage consider the following options:_
>> - [Paperspace](www.paperspace.com) - For under $10 per month, basic cloud compute and storage is available, with automation, Docker containers, and pre-installed Python packages in a Jupyter notebook.
>> - [Google Colab](https://colab.research.google.com/) - Cloud Notebooks with the potential to accelerate with GPUs and TPUs.  Data can be accessed and stored from Google Drive.
>> - [Microsoft Notebooks](https://notebooks.azure.com/) - Cloud Notebooks and data on Azure.
>> - Custom environments: [Amazon Web Services](https://aws.amazon.com/) with EMR, [Microsoft Azure](http://azure.microsoft.com/), [Google Cloud Platform](https://cloud.google.com/gcp), and [IBM Watson Data Studio](https://www.ibm.com/cloud/watson-studio).
>> - Note: Today there are dozens of other platforms that can help in the cloud, including Domino Data Lab, Anaconda Cloud, Crestle, Spell.ai, Comet.ml, among others.

> **2. Developer Environment:** 
>> - Pick a consistent Framework (Python or R) that can be used for your end-to-end project workflow. 
>> - Consider a consistent environment for your project development ([Jupyter](https://jupyter.org/), [PyCharm](https://www.jetbrains.com/pycharm/), or [Visual Studio Code](https://code.visualstudio.com/) which support code, Markdown Text, and LaTeX.

> **3. Data Collection:**
>> - Import your Data in-memory from SQL Databases, [APIs](http://www.programmableweb.com), or Files with [Pandas IO](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) and [Camelot PDFs](https://camelot-py.readthedocs.io/en/master/) or BeautifulSoup for web scraping

> **4. Data Exploration:**
>> - Examine your data, columns and rows and rename and adjust indexing and encoding as appropriate. This [Pandas Cheatsheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf) could be resourcesful for you.  Did you also know that Python has excellent [built-in functions](https://docs.python.org/2/library/functions.html).
>> - Explore null, NaN, None, and missing data with python packages such as [missingno](https://github.com/ResidentMario/missingno) and [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling).  Repair this data either by dropping or imputing values (I.e., mean, median, ffill, bfill, knn calculation)
>> - Indexing: Change indices and datatypes as appropriate for your dataset. (I.e., string, category, integer, float, datetime, timedelta). The datetime module will assist for datetime objects.
>> - Reduce memory constraints: Consider changing datatypes from Int64/Float64 to Int32/16 if memory performance is important for your compute requirements.
>> - Forensically Repair Data: The [regex package](https://bitbucket.org/mrabarnett/mrab-regex/src/default/), or alternatively built-in functions such as .replace and .apply could be used to fix data issues I.e., ($,;|\n\t, etc.)
>> - Repair imbalanced datasets with upsampling or downsampling with imblearn or scikit-learn
>> - Join, Concatenate and merge datasets with Pandas, or SQL modules
>> - Generate statistics for columns, distributions, pivots, and aggregations with Numpy, Scipy, and Pandas modules.
>> - Generate custom calculations, including correlation analysis.
>> - Repair attribute columns with [Pre-processing](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing), [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline), and parameter tuning
>> - List hypothesis for response variable to predict, classify, cluster, or reinforce with Machine Learning.

> **5. Data Visualizations:**
>> - Visualizations can include 100+ types of graphs, available in a sample of the following modules: Turtle, Matplotlib, Seaborn, [Plotly/Dash](https://plot.ly/products/dash/), [Bokeh](https://bokeh.pydata.org/en/latest/), [Altair](https://altair-viz.github.io/), [Plotnine](https://plotnine.readthedocs.io/en/stable/), [Vincent](https://vincent.readthedocs.io/en/latest/), [Mlpd3](http://mpld3.github.io/index.html), [Folium](https://github.com/python-visualization/folium), [pygal](http://pygal.org/en/stable/)), [Sci-kit plot](https://github.com/reiinakano/scikit-plot) and Yellow Brick.
>> - Design considerations: [Color Maps](https://matplotlib.org/examples/color/colormaps_reference.html), [Styles](https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html), and [Palettes](https://seaborn.pydata.org/tutorial/color_palettes.html). Custom colors can be chosen from [Adobe color](https://color.adobe.com/explore/?filter=most-popular&time=month), [Lyft Colorbox](https://www.colorbox.io/), [Geenes](https://geenes.app/user-interface), and [Color Data Styleguides](https://blog.datawrapper.de/colors-for-data-vis-style-guides/).
>> - All graphs/plots must be labeled, formatted and [reproducible](http://www.jesshamrick.com/2016/04/13/reproducible-plots/). All graphs must be saved as PNG files in an Images folder, and saved as an overall PDF for project submission.

### Part 3: Machine Learning Guidelines:
> **Scripts & Notebooks:**
>> - Create Jupyter Notebook or Scripts where DataFrames and data files are called for machine learning pipeline
>> - Revisit your Working Hypothesis(es) to benchmark or backtest your response prediction/classification/cluster/reinforcement. 
>> - Select machine learning modules for your data science (I.e., scikit-learn, statsmodels, pytorch, TensorFlow, Fast.AI, XGBoost, LightGBM, sktime, fbprophet, etc.)
>>> **Note:** _Module versions may require dependencies and may be unstable, and as such, you are recommended to develop and debug in isolated developer environments. (I.e., Conda, Docker, Kubernetes, Cloud instances)_
>> - Perform feature selection/variable importance as a result of Exploratory Data Analysis, Data Visualizations, Dimensionality Reduction, and Model Tuning
>> - Select Algorithms for Machine Learning (I.e., Linear Regression(s), Logistic Regression, Trees, Proximity Models, Classifiers, Natural Language Processors - Spacy, Word2Vec, Time Series, Neural Networks, XGBoost, LightGBM, Ensembles, Stacked Models)
>> - Parameter Tuning - Perform [Grid Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [Randomized Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to optimize parameters for validation, splits, regularization, tuple/dictionary parameters, etc.
>> - Compare and interpret appropriate metrics for regression, classification, or clustering models to the base case/null scenario/benchmark against your majority class.
>> Prepare your model for deployment and [Model Persistance](http://scikit-learn.org/stable/modules/model_persistence.html)

### Part 4: Presentation Design Guidelines:
> _Use this Presentation Skeleton for your Data Science, Solution Engineering or Customer Success Demonstration_

> **1. Cover Slide:**
>> - Project Title/Name 
>> - Team Member Names 
>> - Job Titles, Organizations, E-mail Addresses 

> **2. Agenda Slide:** 
>> - Topics Included and Timing

> **3. Introduction Slide:**
>> - Introduction to your Stakeholders
>> - Introduction to your team 

> **4. Problem Statement Slide:**
>> - Describe Thesis, Problem Statement, or Core Problem 
>> - Describe Solutions both technical/non-technical to the problem 

> **5. Data Analysis Slides:**
>> - Discuss Techniques, Software stack, platforms used
>> - Discuss Data Dictionary, Feature Engineering Techniques 
>> - Discuss Benchmarks or baseline metrics as statistical controls
>> - Discuss data visualizations with business context (Maximum 2 visualizations per slide)

> **6. Machine Learning Slides:**
>> - Discuss the 3 Best Scoring Models or Leaderboard, metrics, and business case interpretation
>> - Describe how robust metrics performed relative to baseline (Model Persistance)

> **7. Model Deployment Slide:**
>> - Discuss how solution will be implemented or Deployed in Production

> **8. Conclusion Slide:**
>> - Abstract of Solution Summary
>> - Recommendations and Results with applied business context
>> - Additional Research and Analysis

> **9. Next Steps Slide:**
>> - Contact information, Github/Gitlab URL, Presentation Link, and Call to Action

> **10. Appendix Slides**
>> - Works Cited and Media Resources 

#### Part 5: Project Submission Guidelines:
> _Submit the following requirements for your project to be considered complete_

> **1. Code Requirements:**
>> - To share with your Advisors on Github, Gitlab, or Bitbucket Repository
>> - To share all code files, Jupyter Notebooks or Script files, data/database files, and digital assets to be shared in a private repository
>> - To Include markdown, LateX, HTML, or Restructured Text to document your Jupyter Notebooks, and to include Comments and Docstrings where relevant for code
>> - To share Final Presentation as PowerPoint AND Adobe PDF
>> - To save and share all graphs and visualizations as separate PNG files in an Images folder, and as a [PDF document](https://stackoverflow.com/questions/17788685/python-saving-multiple-figures-into-one-pdf-file)

> **2. Slides Requirements:**
>> - To focus PowerPoint presentation on applied business use case, analysis, insights, and business impact
>> - To not focus PowerPoint presentation on code
>> - To use less than 3 fonts in Presentation
>> - To include less than 20 slides in Presentation
>> - To present under 7 minutes talking time for Presentation
>> - To practice and prepare for remarks on 3 minutes Questions & Answers for business stakeholders and executive sponsors

## Licenses
License

[![CC-4.0-by-nc-nd](https://licensebuttons.net/l/by-nc-nd/3.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

To the extent possible under law, [David Yakobovitch](http://davidyakobovitch.com/) has licensed this work under Creative Commons, 4.0-NC-ND.  This [license](https://creativecommons.org/licenses/by-nc-nd/4.0/) is the most restrictive of Creative Commons six main licenses, only allowing others to download your works and share them with others as long as they credit the author, but they can’t change them in any way or use them commercially.
