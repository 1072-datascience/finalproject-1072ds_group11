# < Predicting direction of stock price index movement using Trend Deterministic Data >

### Groups
* < name, student ID1 >
* < name, student ID2 >
* < name, student ID3 >
* ...

### Goal
The goal is to predict direction of stock price index movement by financial parameters.

### Demo 
You should provide an example commend to reproduce your result
```R
Rscript code/your_script.R --input data/training --output results/performance.tsv
```
* any on-line visualization

## Folder organization and its related information

### docs
* Your presentation, 1072_datascience_FP_<yourID|groupName>.ppt/pptx/pdf, by **Jun. 25**
* Any related document for the final project
  * papers
  * software user guide

### data

* Source
* Input format
* Any preprocessing?
  * Handle missing data
  * Scale value

### code

* Which method do you use?
  
    knn, logistic regression, naive bayes, random forest, SVM, ANN

* What is a null model for comparison?
  
    We just compare the 6 models.

* How do your perform evaluation? ie. Cross-validation, or extra separated data

    We used stratified random sampling to get 0.2 of  data with the same proportion about stock direction.
    And to use the small models' tuning parameter to train the half of data.

### results

* Which metric do you use 
  * precision, recall, R-square
* Is your improvement significant?
  
    Yes
  
* What is the challenge part of your project?

    
## Reference
* Code/implementation which you include/reference (__You should indicate in your presentation if you use code for others. Otherwise, cheating will result in 0 score for final project.__)

    We studied the paper about predicting stock and stock price in 
    
* Packages you use
    
    
* Related publications
    
    Shiny

