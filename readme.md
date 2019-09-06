<font size="6em"><strong>Data Science Standards</strong></font>
<br>
<strong>What are the Data Science Standards?</strong>
The Data Science Standards are a proven process at both the university and the bootcamp level for students to create production grade machine learning for their portfolio, to excel in the job interview.  This process has been stress-tested with over 2,000 students and offers you the following:

<ul>
<li> A Framework that leads to confidence in job interviews</li>
<li> A portfolio to share with prospective employers</li>
<li> A standard mental model and business framework to solving production grade machine learning </li>
<li> A central repository for state of the art resources for production grade data science</li>
</ul>

<!-- 
Why do I need the Data Science Standards?
Table of Contents
-->

Data science projects require structure.  In essence, you are solving data problems by applying the scientific method.  The following guidelines have been refined with over 2,000 data science students, and is a continuous iteration for building capstone projects for your data science or data engineering portfolio.  If you would like to contribute to the material, you are welcome to share across your considerations, and offer these guidelines with attribution to your colleagues as well.

These guidelines are focused for the Python developer ecosystem.  If you are interested in additional resources, consider some of the following links to explore such as the [distinctions between Python and R](https://www.quora.com/Whats-the-difference-between-machine-learning-in-Python-and-R).  There is a broad scope of resources to learn in Python including these [cheatsheets](https://github.com/chrisallenlane/cheat/tree/master/cheat/cheatsheets) for common integrations with the Python language.  I am confident that with these resources below, your journey will be fruitful for your learning and career.

Other languages: The world of Data Science is rich with algorithms, and multiple languages can support your learning journey.  From Python to C++, you can learn more about the popularity of programming languages with the [Tiobe Index](https://www.tiobe.com/tiobe-index/).

### Part One: Project Proposal Requirements:

Data Science problems are **business problems**.

The field of Big Data has transformed Data Modeling and Data Analytics into Data Science and Data Engineering careers.

One way to approach formulating a question is through goal-setting via the SMART Goals Framework:

1. Specific: The data set and key variables are clearly defined.
2. Measurable: The type of analysis and major assumptions are articulated.
3. Attainable: The question you are asking is feasible for your data set and not likely to be biased.
4. Reproducible: Another person (or future you) can read and understand exactly how your analysis is 5. performed.
6. Time-bound: You clearly state the time period and population to which this analysis pertains.

What Are Some Common Questions Asked in Data Science?
Machine learning more or less asks the following questions:

Does X predict Y? (Where X is a set of data and y is an outcome.)
- Are there any distinct groups in our data?
- What are the key components of our data?
- Is one of our observations “weird”?

From a business perspective, we can ask:
- What is the likelihood that a customer will buy this product?
- Is this a good or bad review?
- How much demand will there be for my service tomorrow?
- Is this the cheapest way to deliver my goods?
- Is there a better way to segment my marketing strategies?
- What groups of products are customers purchasing together?
- Can we automate this simple yes/no decision?

*Note: This list may seem limited, but we rewrite most questions to fit this form.*

**Steps in the Data Science Workflow:**
1. Frame: Develop a hypothesis-driven approach to your analysis.
2. Prepare: Select, import, explore, and clean your data.
3. Analyze: Structure, visualize, and complete your analysis.
4. Interpret: Derive recommendations and business decisions from your data.
5. Communicate: Present (edited) insights from your data to different audiences.

Student Instructions:
1. Brainstorm 5 business problems that you would like to better understand and solve with data science.  
2. Using the above frameworks for SMART and the Data Science Workflow, include these steps in your Project Proposal.

Reading Material:
1. [How to ask Data Science questions](https://towardsdatascience.com/how-to-ask-questions-data-science-can-solve-e073d6a06236)
2. [Asking insightful questions](https://amplitude.com/blog/2015/07/01/question-the-data-how-to-ask-the-right-questions-to-get-actionable-insights)

> After you have identified the project you will work on, then create a markdown (.md) file on Github using the same structure more in-depth, such as an Abstract/Thesis for your project (ideally 300-600 words).  [This is a markdown reference guide.](https://youtu.be/V0fZkWDkPLA)

## Project Proposal Criteria:
> _Please prepare your project proposal as a sharable document, and a PowerPoint/Google Slides presentation_
1. Project Title
- What is your Project Theme?
- What is an Abstract - 1 paragraph Executive Summary of your Solution?
2. Problem Statement & Business Case
- What is the technical problem you are solving?
- What is the applied business case for this problem?> 3. Data Science Workflow 
- What Null/Alternative Hypothesis are you testing against?
- What solutions would you like to deliver against?
- What benchmarks are you looking to automate?
- What alternative questions would you like to explore and provide solutions?
- What analytics and insights would you like to discover from your data? 
- What types of graphics or machine learnings would you like to discover?
- What is the business case for your project? 
- How will your solution help generate revenue, reduce costs, or impact another Key Performance Indicator or Objective Key Result?
- Who will be impacted (Executive Stakeholders/Sponsors) by your solution? Who is your ideal client/customer?
3. Data Collection
- What raw datasets will you extract for machine learning?
- Is the data from open-source, paid crowdsourcing, internal?
- What is the structures, file types, and quality of the data?
- How will you collect the data?
- Of your known data, what is the current data dictionaries that exist, or that you can further describe? (You can create these data dictionaries in a spreadsheet, markdown table, or listed)
4. Data Processing, Preparation, & Feature Engineering
- What techniques will you use to improve your data quality?
- How will you handle missing data and outliers?
- What calculations/formulas would you like to create, that may not yet exist?
5. Machine Learning: Model Selection
- Which model architecture(s) will you use to solve your problem?
- How will you validate the model performance?
6. Model Persistence: Deployment, Training, & Data Pipelines
- How would your results operate LIVE in a production environment?
- What technology stack, what integrations, and which Engineers would you cooperate?
- Where will you share your results internally or externally to stakeholders through Marketing, Implementation and Deployments?
- How will you validate your machine learnings with a timeline from development to production?  How will you generate more data to train?

-------------------------------------------------------------------------------------------------------------------
## Capstone Deliverables:
1. Submit the project proposal as a push to your Github directory with this data science project
2. The project proposal should be submit as a Markdown file ending in .md extension.
3. You can include any types of markdown, LaTeX, text, and images that you find relevant for your project proposal or research abstract.
4. This presentation should be simple, that it can be read and understood by non-technical individuals or business stakeholders.

### Part Two: Exploratory Data Analysis Guidelines
Exploratory data analysis is the first major step in a capstone project for your portfolio.  From collecting and cleaning data, to analyzing and displaying the data, your data wrangling or feature engineering journey will prepare your project for a successful machine learning implementation.
 
> 0. Creating a capstone project can result in stress levels on your machine that cause slow processing power. If you are interested to measure your results consider [timing processing](http://pynash.org/2013/03/06/timing-and-profiling/).  In order to accelerate your prototyping during the development phase, you can consider a cloud solution offering such as [Google Colab](https://colab.research.google.com/) and [importing data into Google Colab](https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory), [Microsoft Notebooks](https://notebooks.azure.com/) and [IBM Notebooks](https://dataplatform.cloud.ibm.com/docs/content/analyze-data/notebooks-parent.html).  If you are looking for more advanced infrastructure, consider providers such as [Amazon Web Services](https://aws.amazon.com/), [Microsoft Azure](http://azure.microsoft.com/), [Google Cloud Platform](https://cloud.google.com/gcp), and [IBM Watson Data Studio](https://www.ibm.com/cloud/watson-studio).  If you are looking for instant container solutions for data science projects, consider [Crestle](https://www.crestle.com/) and [Paperspace](https://www.paperspace.com/)
> 1. To start, please be sure to create Notebooks that you code your data analysis in.  You will want to work in a Python 3 environment. You can also [customize your Jupyter environment](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator) including [adding themes](https://github.com/dunovank/jupyter-themes). If you have legacy python 2 code, a [converter](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/code_prettify/README_2to3.html) does exist.  These notebooks should be through the Jupyter framework, which support .ipynb (iPython Notebooks) and .md (Markdown) files, as well as interactivity between both, which can be displayed effectively through a Github environment or [Binder](https://mybinder.org/). If you would like, you can do additional editing through [VSCode](https://vscodecandothat.com/) and even set it as your [default editor](https://stackoverflow.com/questions/30024353/how-to-use-visual-studio-code-as-default-editor-for-git).  Practing [Jupyter shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) may facilitate your efficiency with the Notebook environment.
> 2. Import your data or [multiple data files](https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe) and to save as dataframes, and convert [XML to DataFrames](http://www.austintaylor.io/lxml/python/pandas/xml/dataframe/2016/07/08/convert-xml-to-pandas-dataframe/) when needed.  And [unzip files](https://chrisjean.com/unzip-multiple-files-from-linux-command-line/) easily.  If you need to scrape data from PDFs consider [Camelot](https://camelot-py.readthedocs.io/en/master/). If you have experience with R, consider the [RPy2 package](https://rpy2.readthedocs.io/en/version_2.8.x/index.html).
> 3. Examine your data, columns and rows and rename and adjust indexing and encoding as appropriate. This [Pandas Cheatsheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf) could be resourcesful for you.  Did you also know that Python has excellent [built-in functions](https://docs.python.org/2/library/functions.html).
> 4. Clean null and blank values, and consider to drop rows, as well as to manipulate data and adjust data types as appropriate, including [dates](https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html) and [time](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html), or setting appropriate indices. Adjusting specific values and replacing strings and characters for the data wrangling process.  
> 5. Explore analysis with graphing and visualizations.  Overall you can view many types of charts [here](https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/?utm_campaign=Data_Elixir&utm_medium=email&utm_source=Data_Elixir_212).  [Here](https://www.anaconda.com/blog/developer-blog/python-data-visualization-2018-why-so-many-libraries/) are all the known packages.  Further, with matplotlib and seaborn and alternative visualization packages ([Plot.ly and Dash](https://plot.ly/products/dash/), [Bokeh](https://bokeh.pydata.org/en/latest/), [Altair](https://altair-viz.github.io/), [Vincent](https://vincent.readthedocs.io/en/latest/), [Mlpd3](http://mpld3.github.io/index.html), [Folium](https://github.com/python-visualization/folium), and [pygal](http://pygal.org/en/stable/)).  It is important to create [reproducible graphs](http://www.jesshamrick.com/2016/04/13/reproducible-plots/). [Sci-kit plot](https://github.com/reiinakano/scikit-plot) may help.  Additional Seaborn resources may be helpful: ([Cat graphs](https://seaborn.pydata.org/generated/seaborn.catplot.html), [Seaborn Color Palettes](https://seaborn.pydata.org/tutorial/color_palettes.html), [Matplotlib Color Maps](https://matplotlib.org/examples/color/colormaps_reference.html) and [more Seaborn examples](https://seaborn.pydata.org/examples/)).  You can also explore [advanced Matplotlib capabilities](https://www.safaribooksonline.com/library/view/python-data-science/9781491912126/ch04.html), [legends with Matplotlib](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html) and [Matplotlib styles](https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html). [Adobe color](https://color.adobe.com/explore/?filter=most-popular&time=month) also offers fantastic color selections and [Lyft Colorbox](https://www.colorbox.io/) provides accessible color options. Numerous [magic methods](https://ipython.readthedocs.io/en/stable/interactive/magics.html) exist to allow graphs to display and to offer [customized magical functions](https://github.com/RafeKettler/magicmethods/blob/master/magicmethods.pdf).
> 6. Perform additional analysis by creating new columns for calculations, including aggregator functions, counts and groupbys. [Scipy](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html) could be helpful for statistical calculations as well.  Consider what [distributions](http://www.math.wm.edu/~leemis/chart/UDR/UDR.html) you might be working with and [all the possibilities](https://en.wikipedia.org/wiki/List_of_probability_distributions).  Consider [GIS in Python](https://automating-gis-processes.github.io/CSC18/lessons/L1/Intro-Python-GIS.html) for geospatial data.
> 7. Encode categorical variables with a variety of techniques through logical conditions, mapping, applying, where clauses, dummy variables, and one hot encoding. Here is [one method to encodage categorical variables](http://benalexkeen.com/mapping-categorical-data-in-pandas/) in Pandas.  When displaying results, consider to [format](https://pyformat.info/) them as well including as [floats](https://stackoverflow.com/questions/6149006/display-a-float-with-two-decimal-places-in-python/6149115).
> 8. Re-run calculations, including crosstabs or pivots, and new graphs to see results 
> 9. Create correlation matrices, [pairplots](https://seaborn.pydata.org/generated/seaborn.pairplot.html), scatterplot matrices, and [heatmaps](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to determine which attributes should be features for your models and which attributes should not.  Design your visualizations with themes such as [pallettes](https://seaborn.pydata.org/tutorial/color_palettes.html). 
> 10. Identify the response variables(s) that you would want to predict/classify/interpret with data science 
> 11. Perform additional feature engineering as necessary, including Min/Max, Normalizaton, Scaling, and additional Pipeline changes that may be beneficial or helpful when you run machine learning.  If you have trouble installing packages, this [environmental variable resource](https://stackoverflow.com/questions/31615322/zsh-conda-pip-installs-command-not-found) may be helpful.
> 12. Merge or concatenate datasets with [Pandas merging](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html), or SQL methods (I.e., [Learning SQL](https://www.quora.com/What-some-of-the-websites-where-I-can-practice-Advance-SQL), [SQL Joins](http://sqlhints.com/tag/cross-join/), [Joins #2](https://stackoverflow.com/questions/38549/what-is-the-difference-between-inner-join-and-outer-join), [Joins #3](https://stackoverflow.com/questions/17759687/cross-join-vs-inner-join-in-sql-server-2008), [SQL Tutorial](https://community.modeanalytics.com/sql/tutorial/introduction-to-sql/), and [Saving Queries](https://stackoverflow.com/questions/31769736/saving-sql-queries-as-sql-text-file) if you have not already, based on common keys or unique items for more in-depth analysis.  Additional SQL resources include the [SQL Cookbook](https://www.amazon.com/SQL-Cookbook-Solutions-Techniques-Developers/dp/0596009763/ref=sr_1_3?ie=UTF8&qid=1548602827&sr=8-3&keywords=sql+cookbook) and [Seven Databases](https://www.amazon.com/Seven-Databases-Weeks-Modern-Movement-ebook/dp/B07CYLX6FD/ref=sr_1_1?ie=UTF8&qid=1548602862&sr=8-1&keywords=seven+database+design).
> 13. Add commenting and markdown throughout the jupyter notebook to explain the interpretation of your results or to comment on code that may not be human readable, and help you recall for you what you are referencing. (Markdown references: [Latex Cheatsheet](https://www.nyu.edu/projects/beber/files/Chang_LaTeX_sheet.pdf), [Markdown for Jupyter Notebooks](https://medium.com/ibm-data-science-experience/markdown-for-jupyter-notebooks-cheatsheet-386c05aeebed), [LaTeX in Notebooks](https://stackoverflow.com/questions/13208286/how-to-write-latex-in-ipython-notebook), [Markdown Intro](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html), [CommonMark](https://commonmark.org/), 
> 14. To create a markdown .md milestone report that shows and explains the results of what you have accomplished to date in this part of your course project. Consider also creating a .pdf or .pptx to display initial results, aha moments, or findings that would be novel or fascinating for your final presentations. 

### Part Three: Machine Learning Guidelines  
> 0. Create a brand new Jupyter notebook, where you run the latest DataFrame or .csv files(s) that you have previously saved from your exploratory data analysis notebook. 
> 1. After you have completed the exploratory data analysis section of your project, start revisiting your hypothesis(es) on ideas that you would like to either predict (regression) or classify (classifier).  > 2. Have you identified a specific column or multiple columns that could be treated as response or target variables to predict/classify?
> 3. If not, consider performing additional exploratory analysis that helps you pinpoint a potential working hypothesis to test results against. You could consider [clustering techniques](http://scikit-learn.org/stable/modules/clustering.html) as an addition to exploratory data analysis as a preparation for machine learning, including [TSNE Clustering](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
> 4. Consider for your machine learning what parts of your feature engineering have been completed, or need to additionally be completed through [Pre-processing](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) and its [use cases](http://scikit-learn.org/stable/modules/preprocessing.html) or [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) operations such as [Normalize](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html), Scaler, Min/Max, etc. 
> 5. As a result of correlation matrices, heatmaps, and visualizations, consider which features may be relevant to support the model that you are building. 
> 6. Consider what machine learning models through [SkLearn](http://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf) and their [Github Repo](https://github.com/scikit-learn/scikit-learn) or [StatsModels](https://www.statsmodels.org/stable/index.html) could be effective for your newly discovered [hypothesis testing](http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-24.html?view=classic) (linear regressions (I.e., [Lowess Regression](http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html), [Logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [multi-class models](http://scikit-learn.org/stable/modules/multiclass.html), KNearest Neighbors, [Clustering](http://scikit-learn.org/stable/modules/clustering.html), [Decision Trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and how to [export graphviz](http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html), including [Bagging Regressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html) or the [Bagging Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html), and [feature selection for Ensembles](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py), [Random Forest](https://victorzhou.com/blog/intro-to-random-forests/) including [Tuning RF](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74), Naive Bayes, Natural Language Processing ([Word2Vec](https://github.com/davidyakobovitch/word2vec-translation) and [understanding Word2Vec](https://jalammar.github.io/illustrated-word2vec/), [Spacy](https://github.com/davidyakobovitch/spaCy-tutorial) and [Spacy Models](https://spacy.io/usage/models), and [Topic Modeling](https://github.com/davidyakobovitch/topic-modeling)) Time Series Analysis (I.e., [Time Series 1](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/) and [Time Series 2](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/), [Neural Networks](http://scikit-learn.org/stable/modules/neural_networks_supervised.html), Support Vector Machines and [Model Resistance](http://scikit-learn.org/stable/modules/model_persistence.html), [Stochastic Gradient Descent](http://www.scikit-learn.org/stable/modules/sgd.html), dimensionality reduction with PCA (<a href="http://setosa.io/ev/principal-component-analysis/">demo here</a>) as well as Ensembles such as [GB Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) and [GB Regressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)).  Once you have determined models to consider, be sure to import their packages into Python.
> 7. Consider what tuning parameters you may want to optimize for your model, including regularization (Lasso, ridge, ElasticNet), and additional parameters relevant to each model.  [Github Code Search](http://jakubdziworski.github.io/tools/2016/08/26/github-code-advances-search-programmers-goldmine.html) could help you as you are adjusting your models.
> 8.  Be sure to include a train_test_split, and then consider a KFolds or Cross Validations to offer stratified results that limit the interpretation of outliers for your dataset.  If you have imbalanced classes consider [techniques to adjust them](https://elitedatascience.com/imbalanced-classes).
> 9. If you still have many outliers, consider how to remove them or optimize for them with categories.  How could you adjust your categories, or thresholds to improve performance for what you are testing for your hypothesis? Depending on how your model error performs, you may want to consider to change or adjust other features in your model.  You may want to consider to add or remove features, and measure the feature importance when running models. 
> 10.  Consider a [Grid Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [Grid Search with Cross Validation Continued](http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html), or [Random Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to better optimize your models. 
> 11.  Share metrics on each model that is run, such as error and accuracy, confusion matrices which are based off [truth tables](https://en.wikipedia.org/wiki/Truth_table), and [logical conditions](https://en.wikipedia.org/wiki/Sensitivity_and_specificity). They can be displayed through [ROC/AUC](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) curves as well as [visually](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py). [Scoring your models](http://benalexkeen.com/scoring-classifier-models-using-scikit-learn/) is important for both regression and classification techniques.  Other models have additional metrics, that you can consider to share.  You can set up metrics and running models in defined functions for further automation of your project. 
> 12. Compare your metrics against the base case or null case for accuracy, which ideally is compared to your majority class, or a median/mean representation for your target/response variable.  How well does your model perform?
> 13. Provide markdown explaining the interpretation relevant to your business case after running models.  Also, share comments to explain what you are doing, for your interpretation and then reproducibility of your code. 
> 14.  If you are running Time Series Analysis, you will want to consider additional model capabilities such as rolling and moving averages with the dateTime package and pandas.
> 15.  If you are working on Natural Language processing, you will want to consider python packages such as [Spacy](https://github.com/davidyakobovitch/spaCy-tutorial), [topic modeling](https://github.com/davidyakobovitch/topic-modeling),  NLTK, TextBlob, and [word2vec](https://github.com/davidyakobovitch/word2vec-translation).
> 16. If you are scraping additional data, consider python packages such as Selenium and BeautifulSoup4.
> 17.  For your project, your presentation will showcase the best 3-5 models.  However, it is fine if you have inefficient models that do not perform well, for practice, so keep these in your main modeling Jupyter notebook as a reference. 
> 18. If you chose to work with .py scripts, here is a [method to rename these files](https://stackoverflow.com/questions/2759067/rename-multiple-files-in-a-directory-in-python/24954254).

### Part Four: Presentation Design
#### Presentation Skeleton for Data Science, Solution Engineering & Customer Experience

- Title Page:
  - Project Title 
  - Name 
  - Job Title, Organizational Title 
- Agenda Page: 
  - Sections to be covered, and time for each section 
- Introductions: 
  - Introduction to your Stakeholders
  - Introduction to you 
- Problem Statement:
  - Describe in Depth the Problem 
  - Solution(s) technical/non-technical to the problem 
- Data Analysis Slide(s):
  - Techniques, Software stack, platforms used 
  - Data Dictionary, Feature Engineering
  - Benchmarked metrics to discover
  - Visualizations of analytics with business context 
- Machine Learning Slide(s):
  - Metrics and Scoring with analytics with best scoring models and business context 
  - Describe how metrics are scored to baseline 
- Deployment:
  - How Machine Learning solution will be Deployed
- Conclusion Slide:
  - Recommendations and Results with business context
  - Future Research and Analysis 
- Next Steps slide:
  - Contact, Github URL, Presentation Link, and Call to Action 
 Appendix:
  - Works Cited and Media Resources 

#### Design and Product Requirements
<ol>
<li> Github Organization: Create one parent directory for your project, with separate Jupyter Notebooks for each section, a data folder, and an assets folder for images.</li>
<li> Final presentation to be shared as a Google Slides presentation or Microsoft PowerPoint or React Native Slides</li>
<li> Presentation to focus on business analysis, insights, and business impact with graphs, and machine learning output.  Minimal, if any, code should be shown in presentation.</li> 
<li> Presentation should use maximum of 3 fonts.</li>
<li> Maximum of 20 slides.</li>
<li> Can be interpreted if sent as a cold e-mail without you presenting your report.</li>
<li> Appendix Slide for Works Cited, Bibliography, and Links must be included.</li>
<li> Presentation delivery to not exceed 8 minutes </li>
<li> Presentation delivery to be for non-technical stakeholder (Also known as "Teach me like I am 5")</li>
<li> Presentation Delivery on Google Hangouts, Skype, or Zoom </li>
<li> Be prepared for a Q&A sesssion for 3 to 5 minutes</li>
</ol>

#### Additional Notes
> 1. Consider that you could save all your plots to [an overall PDF](https://stackoverflow.com/questions/17788685/python-saving-multiple-figures-into-one-pdf-file).
> 2. You could consider a pass-through on your Jupyter notebooks to customize them with [docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and Markdown to polish your presentation for code review by stakeholders.

## Licenses
License

[![CC-4.0-by-nc-nd](https://licensebuttons.net/l/by-nc-nd/3.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

To the extent possible under law, [David Yakobovitch](http://davidyakobovitch.com/) has licensed this work under Creative Commons, 4.0-NC-ND.  This [license](https://creativecommons.org/licenses/by-nc-nd/4.0/) is the most restrictive of Creative Commons six main licenses, only allowing others to download your works and share them with others as long as they credit the author, but they can’t change them in any way or use them commercially.
