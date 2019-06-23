### packages
library(randomForest)
library(neuralnet)
library(nnet)
library(caret)
library(kernlab)
library(e1071)
library(naivebayes)

### load data and add and adjust new variables
d <- read.csv('data/JPY1.csv', header = T)
d$year <- factor(substr(d$Date, 1, 4))
d$dir <- factor(c(NA,ifelse(tail(d$Close,-1)-head(d$Close,-1)>=0,1,0)))
vars <- setdiff(colnames(d), c('Date','Date.1','Open','High','Low','Close','dir','MACDsignal','MACDhist','year'))
norm_vars <- c()
for (i in vars){
  name <- paste0('norm_',i)
  d[name] <- (d[,i]-min(d[,i],na.rm = TRUE))/(max(d[,i],na.rm = TRUE)-min(d[,i],na.rm = TRUE))
  norm_vars <- c(norm_vars,name)
}

### stratified random sampling  to get 0.2 of all data with year and balance 0 / 1 to train a model
set.seed(777)
for (y in setdiff(levels(d$year),'2000')){
  index_sample1 <- sample(rownames(subset(d,year==y & dir==1)),round(0.2*length(rownames(subset(d,year==y & dir==1)))))
  index_sample0 <- sample(rownames(subset(d,year==y & dir==0)),round(0.2*length(rownames(subset(d,year==y & dir==0)))))
  if (y=='2001'){
    d_parameter_selection <- d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('year','dir',norm_vars)]
  }else{
    d_parameter_selection <- rbind(d_parameter_selection,d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('year','dir',norm_vars)])
  }
}

### straified random sampling to get the half of all data with year and balance 0 / 1
for (y in setdiff(levels(d$year),'2000')){
  index_sample1 <- sample(rownames(subset(d_parameter_selection,year==y & dir==1)),round(0.5*dim(subset(d_parameter_selection,year==y & dir==1)))[1])
  index_sample0 <- sample(rownames(subset(d_parameter_selection,year==y & dir==0)),round(0.5*dim(subset(d_parameter_selection,year==y & dir==0)))[1])
  if (y=='2001'){
    d_ps_train <- d_parameter_selection[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars)]
    d_ps_holdout <- d_parameter_selection[setdiff(rownames(subset(d_parameter_selection,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE))),c('dir',norm_vars)]
  }else{
    d_ps_train <- rbind(d_ps_train,d_parameter_selection[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars)])
    d_ps_holdout <- rbind(d_ps_holdout,d_parameter_selection[setdiff(rownames(subset(d_parameter_selection,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE))),c('dir',norm_vars)])
  }
}


### train model with stratified random sampling data
# train ANN model and record tuning parameters
ann_ps_pred_train <- data.frame(n=factor(),accuracy_train=factor())
ann_ps_pred_holdout <- data.frame(n=factor(),accuracy_holdout=factor())
for (n in seq(10,100,10)){
  ann_model <- neuralnet(dir=='1' ~., data=d_ps_train, hidden=n, learningrate=0.1,linear.output = FALSE,stepmax=1e+06)
  ann_pred_train <- ifelse(predict(ann_model,newdata = d_ps_train[,norm_vars])>=0.5,1,0)
  ann_pred_holdout <- ifelse(predict(ann_model,newdata = d_ps_holdout[,norm_vars])>=0.5,1,0)
  ann_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=ann_pred_train)
  ann_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=ann_pred_holdout)
  ann_ps_pred_train <- rbind(ann_ps_pred_train,data.frame(n=n,accuracy_train=sum(diag(table(ann_result_train))/dim(d_ps_train)[1])))
  ann_ps_pred_holdout <- rbind(ann_ps_pred_holdout,data.frame(n=n,accuracy_holdout=sum(diag(table(ann_result_holdout))/dim(d_ps_holdout)[1])))
  ann_ps_pred <- merge(ann_ps_pred_train,ann_ps_pred_holdout,by='n')
  ann_ps_pred$mean <- (ann_ps_pred$accuracy_train+ann_ps_pred$accuracy_holdout)/2
}
ann_parameter <- ann_ps_pred[order(ann_ps_pred$mean,decreasing = TRUE),][1:3,1]

# train SVM model and record tuning parameters
svm_ps_pred_train <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_train=factor())
svm_ps_pred_holdout <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_holdout=factor())
for (kernel in c('polydot','rbfdot')){
  if(kernel=='polydot'){
    for (degree in 1:4){
      for (c in c(0.5,1,5,10,100)){
        svm_model <- ksvm(dir~., data=d_ps_train, kernel=kernel, degree=degree, C=c)
        svm_pred_train <- predict(svm_model,newdata = d_ps_train)
        svm_pred_holdout <- predict(svm_model,newdata = d_ps_holdout)
        svm_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=svm_pred_train)
        svm_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=svm_pred_holdout)
        svm_ps_pred_train <- rbind(svm_ps_pred_train,data.frame(kernel=kernel,degree=degree,gamma=NA,c=c,accuracy_train=sum(diag(table(svm_result_train))/dim(d_ps_train)[1])))
        svm_ps_pred_holdout <- rbind(svm_ps_pred_holdout,data.frame(kernel=kernel,degree=degree,gamma=NA,c=c,accuracy_holdout=sum(diag(table(svm_result_holdout))/dim(d_ps_holdout)[1])))
      }
    }
  } else if(kernel=='rbfdot'){
    for (gamma in c(seq(0.5,5,0.5),10)){
      for (c in c(0.5,1,5,10)){
        svm_model <- ksvm(dir~., data=d_ps_train, kernel=kernel, sigma=gamma, C=c)
        svm_pred_train <- predict(svm_model,newdata = d_ps_train)
        svm_pred_holdout <- predict(svm_model,newdata = d_ps_holdout)
        svm_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=svm_pred_train)
        svm_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=svm_pred_holdout)
        svm_ps_pred_train <- rbind(svm_ps_pred_train,data.frame(kernel=kernel,degree=NA,gamma=gamma,c=c,accuracy_train=sum(diag(table(svm_result_train))/dim(d_ps_train)[1])))
        svm_ps_pred_holdout <- rbind(svm_ps_pred_holdout,data.frame(kernel=kernel,degree=NA,gamma=gamma,c=c,accuracy_holdout=sum(diag(table(svm_result_holdout))/dim(d_ps_holdout)[1])))
      }
    }
  }
  svm_ps_pred <- merge(svm_ps_pred_train,svm_ps_pred_holdout,by=c('kernel','degree','gamma','c'))
  svm_ps_pred$mean <- (svm_ps_pred$accuracy_train+svm_ps_pred$accuracy_holdout)/2
}
svm_parameter <- rbind(subset(svm_ps_pred,kernel=='polydot')[order(subset(svm_ps_pred,kernel=='polydot')$mean,decreasing = TRUE),][1,1:4],subset(svm_ps_pred,kernel=='rbfdot')[order(subset(svm_ps_pred,kernel=='rbfdot')$mean,decreasing = TRUE),][1,1:4])

# train random forest model and record tuning parameters
rf_ps_pred_train <- data.frame(ntree=factor(),accuracy_train=factor())
rf_ps_pred_holdout <- data.frame(ntree=factor(),accuracy_holdout=factor())
for (i in seq(10,200,10)){
  rf_model <- randomForest(dir~., data=d_ps_train, ntree=i,importance = T)
  rf_pred_train <- predict(rf_model,newdata = d_ps_train)
  rf_pred_holdout <- predict(rf_model,newdata = d_ps_holdout)
  rf_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=rf_pred_train)
  rf_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=rf_pred_holdout)
  rf_ps_pred_train <- rbind(rf_ps_pred_train,data.frame(ntree=i,accuracy_train=sum(diag(table(rf_result_train))/dim(d_ps_train)[1])))
  rf_ps_pred_holdout <- rbind(rf_ps_pred_holdout,data.frame(ntree=i,accuracy_holdout=sum(diag(table(rf_result_holdout))/dim(d_ps_holdout)[1])))
  rf_ps_pred <- merge(rf_ps_pred_train,rf_ps_pred_holdout,by='ntree')
  rf_ps_pred$mean <- (rf_ps_pred$accuracy_train+rf_ps_pred$accuracy_holdout)/2
}
rf_parameter <- rf_ps_pred[order(rf_ps_pred$mean,decreasing = TRUE),][1:3,1]

# train Naive Bayes model and record tuning parameters
NB_model = naive_bayes(dir ~ ., usekernel = T, data = d_ps_train)
train_predict = predict(NB_model, d_ps_train, type = "class")
table(d_ps_train$dir, train_predict)
mean(d_ps_train$dir == train_predict)

holdout_predict = predict(NB_model, d_ps_holdout, type = "class")
table(d_ps_holdout$dir, holdout_predict)
mean(d_ps_holdout$dir == holdout_predict)

NB_model = naive_bayes(dir ~ ., usekernel = T, data = d_ps_train)

# Comparison of these models
for (y in setdiff(levels(d$year),'2000')){
  index_sample1 <- sample(rownames(subset(d,year==y & dir==1)),round(0.5*length(which(d$year==y & d$dir==1))))
  index_sample0 <- sample(rownames(subset(d,year==y & dir==0)),round(0.5*length(which(d$year==y & d$dir==0))))
  if (y=='2001'){
    d_train <- d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars)]
    d_holdout <- d[setdiff(rownames(subset(d,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE),rownames(subset(d,year=='2000')))),c('dir',norm_vars)]
  }else{
    d_train <- rbind(d_train,d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars)])
    d_holdout <- rbind(d_holdout,d[setdiff(rownames(subset(d,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE),rownames(subset(d,year=='2000')))),c('dir',norm_vars)])
  }
}

### using the stratified model's tuning parameters to build the same tuning parameters model for the bigger train datasets.
#ANN
ann_performance <- data.frame(n=factor(),accuracy_train=factor(),accuracy_holdout=factor())
for (n in ann_parameter){
  ann_model <- neuralnet(dir=='1'~., data=d_train, hidden=n,learningrate = 0.1,linear.output = FALSE,threshold = 0.01,stepmax = 1e+06)
  ann_pred_train <- ifelse(predict(ann_model,newdata = d_train[,norm_vars])>=0.5,1,0)
  ann_pred_holdout <- ifelse(predict(ann_model,newdata = d_holdout[,norm_vars])>=0.5,1,0)
  ann_result_train <- data.frame(dir=d_train[,'dir'],pred=ann_pred_train)
  ann_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=ann_pred_holdout)
  accuracy_train <- sum(diag(table(ann_result_train))/dim(d_train)[1])
  accuracy_holdout <- sum(diag(table(ann_result_holdout))/dim(d_holdout)[1])
  ann_performance <- rbind(ann_performance,data.frame(n=n,accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
}
ann_performance$mean <- (as.numeric(ann_performance$accuracy_train)+as.numeric(ann_performance$accuracy_holdout))/2

#SVM
svm_performance <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_train=factor(),accuracy_holdout=factor())
svm_model <- ksvm(dir~., data=d_train, kernel=as.character(svm_parameter[1,1]), degree=as.integer(svm_parameter[1,2]), C=as.integer(svm_parameter[1,4]))
svm_pred_train <- predict(svm_model,newdata = d_train)
svm_pred_holdout <- predict(svm_model,newdata = d_holdout)
svm_result_train <- data.frame(dir=d_train[,'dir'],pred=svm_pred_train)
svm_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=svm_pred_holdout)
accuracy_train <- sum(diag(table(svm_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(svm_result_holdout))/dim(d_holdout)[1])
svm_performance <- rbind(svm_performance,data.frame(kernel=svm_parameter[1,1],degree=svm_parameter[1,2],gamma=NA,c=svm_parameter[1,4],accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
svm_model <- ksvm(dir~., data=d_train, kernel=as.character(svm_parameter[2,1]), sigma=as.integer(svm_parameter[2,3]), C=as.integer(svm_parameter[2,4]))
svm_pred_train <- predict(svm_model,newdata = d_train)
svm_pred_holdout <- predict(svm_model,newdata = d_holdout)
svm_result_train <- data.frame(dir=d_train[,'dir'],pred=svm_pred_train)
svm_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=svm_pred_holdout)
accuracy_train <- sum(diag(table(svm_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(svm_result_holdout))/dim(d_holdout)[1])
svm_performance <- rbind(svm_performance,data.frame(kernel=svm_parameter[2,1],degree=NA,gamma=svm_parameter[2,3],c=svm_parameter[2,4],accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
svm_performance$mean <- (as.numeric(svm_performance$accuracy_train)+as.numeric(svm_performance$accuracy_holdout))/2

#randomForest
rf_performance <- data.frame(ntree=factor(),accuracy_train=factor(),accuracy_holdout=factor())
for (n in rf_parameter){
  rf_model <- randomForest(dir~., data=d_train, ntree=i,importance = T)
  rf_pred_train <- predict(rf_model,newdata = d_train)
  rf_pred_holdout <- predict(rf_model,newdata = d_holdout)
  rf_result_train <- data.frame(dir=d_train[,'dir'],pred=rf_pred_train)
  rf_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=rf_pred_holdout)
  accuracy_train <- sum(diag(table(rf_result_train))/dim(d_train)[1])
  accuracy_holdout <- sum(diag(table(rf_result_holdout))/dim(d_holdout)[1])
  rf_performance <- rbind(rf_performance,data.frame(ntree=n,accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
}
rf_performance$mean <- (as.numeric(rf_performance$accuracy_train)+as.numeric(rf_performance$accuracy_holdout))/2

#table
compare_list <- list(ANN=ann_performance,SVM=svm_performance,RandomForest=rf_performance)

### trials
## continuous
# logistic regression 0.8
model = glm(dir ~ ., data = d_ps_train[1:11], family = binomial(link="logit"), na.action=na.exclude)
holdout_predict = ifelse(predict(model, d_ps_holdout, type = "response") >= 0.5, 1, 0)
table(d_ps_holdout$dir, holdout_predict)
mean(d_ps_holdout$dir == holdout_predict)

# knn 0.7
library(class)
hold_knn = knn(d_ps_train[2:11], d_ps_holdout[2:11], d_ps_train$dir, k = 5)
table(d_ps_holdout$dir, hold_knn)
mean(d_ps_holdout$dir == hold_knn)

## discrete
# logistic regression 0.95
model = glm(dir ~ ., data = d_ps_train[,c(1,12:20)], family = binomial(link="logit"), na.action=na.exclude)
holdout_predict = ifelse(predict(model, d_ps_holdout, type = "response") >= 0.5, 1, 0)
table(d_ps_holdout$dir, holdout_predict)
mean(d_ps_holdout$dir == holdout_predict)

# knn 0.93
library(class)
hold_knn = knn(d_ps_train[12:20], d_ps_holdout[12:20], d_ps_train$dir, k = 5)
table(d_ps_holdout$dir, hold_knn)
mean(d_ps_holdout$dir == hold_knn)

## half data
# logistic regression 0.8
model = glm(dir ~ ., data = d_train[1:11], family = binomial(link="logit"), na.action=na.exclude)
holdout_predict = ifelse(predict(model, d_holdout, type = "response") >= 0.5, 1, 0)
table(d_holdout$dir, holdout_predict)
mean(d_holdout$dir == holdout_predict)

# knn 0.7
library(class)
hold_knn = knn(d_train[2:11], d_holdout[2:11], d_train$dir, k = 5)
table(d_holdout$dir, hold_knn)
mean(d_holdout$dir == hold_knn)

## discrete
# logistic regression 0.95
model = glm(dir ~ ., data = d_train[,c(1,12:20)], family = binomial(link="logit"), na.action=na.exclude)
holdout_predict = ifelse(predict(model, d_holdout, type = "response") >= 0.5, 1, 0)
table(d_holdout$dir, holdout_predict)
mean(d_holdout$dir == holdout_predict)

# knn 0.93
library(class)
knn_accuracy = matrix(0L, nrow = 20, ncol = 1)
for(k in 1:20){
  hold_knn = knn(d_train[12:20], d_holdout[12:20], d_train$dir, k = k)
  knn_accuracy[k,1] = round(mean(d_holdout$dir == hold_knn), digits = 4)
}
table(d_holdout$dir, hold_knn)
mean(d_holdout$dir == hold_knn)


### pca
library(ggbiplot)
d_train$dir = as.numeric(as.character(d_train$dir))
d_train_pca = prcomp(d_train[1:11], center = TRUE, scale. = TRUE)
ggbiplot(d_train_pca, obs.scale = 1, var.scale = 1, groups = d_train$dir, varname.size = 0, circle = TRUE, ellipse = TRUE)
