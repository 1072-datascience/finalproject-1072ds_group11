#packages
library(randomForest)
library(neuralnet)
library(nnet)
library(caret)
library(kernlab)
library(e1071)
library(naivebayes)

#load data and add and adjust new variables
d <- read.csv('/Users/lucas/Downloads/JPY1.csv', header = T)
d$year <- factor(substr(d$Date, 1, 4))
d$dir <- factor(c(NA,ifelse(tail(d$Close,-1)-head(d$Close,-1)>=0,1,0)))
vars <- setdiff(colnames(d), c('Date','Date.1','Open','High','Low','Close','dir','MACDsignal','MACDhist','year'))
#normalise variables
norm_vars <- c()
for (i in vars){
  name <- paste0('norm_',i)
  d[name] <- (2*d[,i]-(max(d[,i],na.rm = TRUE)+min(d[,i],na.rm = TRUE)))/(max(d[,i],na.rm = TRUE)-min(d[,i],na.rm = TRUE))
  norm_vars <- c(norm_vars,name)
}
#discretise variables
dc_vars <- c()
d$dc_SMA <- ifelse(d$Close-d$SMA>=0,1,-1)
d$dc_WMA <- ifelse(d$Close-d$WMA>=0,1,-1)
d$dc_K <- c(NA,ifelse(tail(d$K,-1)-head(d$K,-1)>=0,1,-1))
d$dc_D <- c(NA,ifelse(tail(d$D,-1)-head(d$D,-1)>=0,1,-1))
d$dc_R <- c(NA,ifelse(tail(d$R,-1)-head(d$R,-1)>=0,1,-1))
d$dc_MACD <- c(NA,ifelse(tail(d$MACD,-1)-head(d$MACD,-1)>=0,1,-1))
d$dc_RSI <- ifelse(d$RSI>70,-1,ifelse(d$RSI<30,1,ifelse(c(NA,tail(d$RSI,-1)-head(d$RSI,-1))>=0,1,-1)))
d$dc_CCI <- ifelse(d$CCI>200,-1,ifelse(d$CCI<-200,1,ifelse(c(NA,tail(d$CCI,-1)-head(d$CCI,-1))>=0,1,-1)))
d$dc_MOMENTUM <- c(NA,ifelse(tail(d$MOMENTUM,-1)-head(d$MOMENTUM,-1)>=0,1,-1))
for (i in setdiff(c(vars),'ADX')){
  name <- paste0('dc_',i)
  dc_vars <- c(dc_vars,name)
}
#parameter selection data
for (y in setdiff(levels(d$year),'2000')){
  index_sample1 <- sample(rownames(subset(d,year==y & dir==1)),round(0.2*length(rownames(subset(d,year==y & dir==1)))))
  index_sample0 <- sample(rownames(subset(d,year==y & dir==0)),round(0.2*length(rownames(subset(d,year==y & dir==0)))))
  if (y=='2001'){
    d_parameter_selection <- d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('year','dir',norm_vars,dc_vars)]
  }else{
    d_parameter_selection <- rbind(d_parameter_selection,d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('year','dir',norm_vars,dc_vars)])
  }
}
for (y in setdiff(levels(d$year),'2000')){
  index_sample1 <- sample(rownames(subset(d_parameter_selection,year==y & dir==1)),round(0.5*dim(subset(d_parameter_selection,year==y & dir==1)))[1])
  index_sample0 <- sample(rownames(subset(d_parameter_selection,year==y & dir==0)),round(0.5*dim(subset(d_parameter_selection,year==y & dir==0)))[1])
  if (y=='2001'){
    d_ps_train <- d_parameter_selection[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars,dc_vars)]
    d_ps_holdout <- d_parameter_selection[setdiff(rownames(subset(d_parameter_selection,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE))),c('dir',norm_vars,dc_vars)]
  }else{
    d_ps_train <- rbind(d_ps_train,d_parameter_selection[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars,dc_vars)])
    d_ps_holdout <- rbind(d_ps_holdout,d_parameter_selection[setdiff(rownames(subset(d_parameter_selection,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE))),c('dir',norm_vars,dc_vars)])
  }
}
#data for comparison
for (y in setdiff(levels(d$year),'2000')){
  index_sample1 <- sample(rownames(subset(d,year==y & dir==1)),round(0.5*length(which(d$year==y & d$dir==1))))
  index_sample0 <- sample(rownames(subset(d,year==y & dir==0)),round(0.5*length(which(d$year==y & d$dir==0))))
  if (y=='2001'){
    d_train <- d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars,dc_vars)]
    d_holdout <- d[setdiff(rownames(subset(d,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE),rownames(subset(d,year=='2000')))),c('dir',norm_vars,dc_vars)]
  }else{
    d_train <- rbind(d_train,d[sort(c(index_sample1,index_sample0),decreasing = FALSE),c('dir',norm_vars,dc_vars)])
    d_holdout <- rbind(d_holdout,d[setdiff(rownames(subset(d,year==y)),c(sort(c(index_sample1,index_sample0),decreasing = FALSE),rownames(subset(d,year=='2000')))),c('dir',norm_vars,dc_vars)])
  }
}
#PARTI: normalised data
#choose ANN parameters
ann_ps_pred_train <- data.frame(n=factor(),accuracy_train=factor())
ann_ps_pred_holdout <- data.frame(n=factor(),accuracy_holdout=factor())
for (n in seq(10,100,10)){
  ann_model <- neuralnet(dir=='1' ~., data=d_ps_train[,c('dir',norm_vars)], hidden=n, learningrate=0.1,linear.output = FALSE,stepmax=1e+06)
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
#choose SVM parameters
svm_ps_pred_train <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_train=factor())
svm_ps_pred_holdout <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_holdout=factor())
for (kernel in c('polydot','rbfdot')){
  if(kernel=='polydot'){
    for (degree in 1:4){
      for (c in c(0.5,1,5,10,100)){
        svm_model <- ksvm(dir~., data=d_ps_train[,c('dir',norm_vars)], kernel=kernel, degree=degree, C=c)
        svm_pred_train <- predict(svm_model,newdata = d_ps_train[,c('dir',norm_vars)])
        svm_pred_holdout <- predict(svm_model,newdata = d_ps_holdout[,c('dir',norm_vars)])
        svm_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=svm_pred_train)
        svm_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=svm_pred_holdout)
        svm_ps_pred_train <- rbind(svm_ps_pred_train,data.frame(kernel=kernel,degree=degree,gamma=NA,c=c,accuracy_train=sum(diag(table(svm_result_train))/dim(d_ps_train)[1])))
        svm_ps_pred_holdout <- rbind(svm_ps_pred_holdout,data.frame(kernel=kernel,degree=degree,gamma=NA,c=c,accuracy_holdout=sum(diag(table(svm_result_holdout))/dim(d_ps_holdout)[1])))
      }
    }
  } else if(kernel=='rbfdot'){
    for (gamma in c(seq(0.5,5,0.5),10)){
      for (c in c(0.5,1,5,10)){
        svm_model <- ksvm(dir~., data=d_ps_train[,c('dir',norm_vars)], kernel=kernel, sigma=gamma, C=c)
        svm_pred_train <- predict(svm_model,newdata = d_ps_train[,c('dir',norm_vars)])
        svm_pred_holdout <- predict(svm_model,newdata = d_ps_holdout[,c('dir',norm_vars)])
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
#choose randomForest parameters
rf_ps_pred_train <- data.frame(ntree=factor(),accuracy_train=factor())
rf_ps_pred_holdout <- data.frame(ntree=factor(),accuracy_holdout=factor())
for (i in seq(10,200,10)){
  rf_model <- randomForest(dir~., data=d_ps_train[,c('dir',norm_vars)], ntree=i,importance = T)
  rf_pred_train <- predict(rf_model,newdata = d_ps_train[,c('dir',norm_vars)])
  rf_pred_holdout <- predict(rf_model,newdata = d_ps_holdout[,c('dir',norm_vars)])
  rf_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=rf_pred_train)
  rf_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=rf_pred_holdout)
  rf_ps_pred_train <- rbind(rf_ps_pred_train,data.frame(ntree=i,accuracy_train=sum(diag(table(rf_result_train))/dim(d_ps_train)[1])))
  rf_ps_pred_holdout <- rbind(rf_ps_pred_holdout,data.frame(ntree=i,accuracy_holdout=sum(diag(table(rf_result_holdout))/dim(d_ps_holdout)[1])))
  rf_ps_pred <- merge(rf_ps_pred_train,rf_ps_pred_holdout,by='ntree')
  rf_ps_pred$mean <- (rf_ps_pred$accuracy_train+rf_ps_pred$accuracy_holdout)/2
}
rf_parameter <- rf_ps_pred[order(rf_ps_pred$mean,decreasing = TRUE),][1:3,1]
#ANN
ann_performance <- data.frame(n=factor(),accuracy_train=factor(),accuracy_holdout=factor())
for (n in ann_parameter){
  ann_model <- neuralnet(dir=='1'~., data=d_train[,c('dir',norm_vars)], hidden=n,learningrate = 0.1,linear.output = FALSE,threshold = 0.01,stepmax = 1e+06)
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
svm_model <- ksvm(dir~., data=d_train[,c('dir',norm_vars)], kernel=as.character(svm_parameter[1,1]), degree=as.numeric(svm_parameter[1,2]), C=as.numeric(svm_parameter[1,4]))
svm_pred_train <- predict(svm_model,newdata = d_train[,c('dir',norm_vars)])
svm_pred_holdout <- predict(svm_model,newdata = d_holdout[,c('dir',norm_vars)])
svm_result_train <- data.frame(dir=d_train[,'dir'],pred=svm_pred_train)
svm_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=svm_pred_holdout)
accuracy_train <- sum(diag(table(svm_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(svm_result_holdout))/dim(d_holdout)[1])
svm_performance <- rbind(svm_performance,data.frame(kernel=svm_parameter[1,1],degree=svm_parameter[1,2],gamma=NA,c=svm_parameter[1,4],accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
svm_model <- ksvm(dir~., data=d_train[,c('dir',norm_vars)], kernel=as.character(svm_parameter[2,1]), sigma=as.numeric(svm_parameter[2,3]), C=as.numeric(svm_parameter[2,4]))
svm_pred_train <- predict(svm_model,newdata = d_train[,c('dir',norm_vars)])
svm_pred_holdout <- predict(svm_model,newdata = d_holdout[,c('dir',norm_vars)])
svm_result_train <- data.frame(dir=d_train[,'dir'],pred=svm_pred_train)
svm_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=svm_pred_holdout)
accuracy_train <- sum(diag(table(svm_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(svm_result_holdout))/dim(d_holdout)[1])
svm_performance <- rbind(svm_performance,data.frame(kernel=svm_parameter[2,1],degree=NA,gamma=svm_parameter[2,3],c=svm_parameter[2,4],accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
svm_performance$mean <- (as.numeric(svm_performance$accuracy_train)+as.numeric(svm_performance$accuracy_holdout))/2
#randomForest
rf_performance <- data.frame(ntree=factor(),accuracy_train=factor(),accuracy_holdout=factor())
for (n in rf_parameter){
  rf_model <- randomForest(dir~., data=d_train[,c('dir',norm_vars)], ntree=i,importance = T)
  rf_pred_train <- predict(rf_model,newdata = d_train[,c('dir',norm_vars)])
  rf_pred_holdout <- predict(rf_model,newdata = d_holdout[,c('dir',norm_vars)])
  rf_result_train <- data.frame(dir=d_train[,'dir'],pred=rf_pred_train)
  rf_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=rf_pred_holdout)
  accuracy_train <- sum(diag(table(rf_result_train))/dim(d_train)[1])
  accuracy_holdout <- sum(diag(table(rf_result_holdout))/dim(d_holdout)[1])
  rf_performance <- rbind(rf_performance,data.frame(ntree=n,accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
}
rf_performance$mean <- (as.numeric(rf_performance$accuracy_train)+as.numeric(rf_performance$accuracy_holdout))/2
#naive Bayes
NB_model <- naive_bayes(dir ~ ., data = d_train[,c('dir',norm_vars)])
NB_pred_train <- predict(NB_model, newdata = d_train[,norm_vars])
NB_pred_holdout <- predict(NB_model, newdata = d_holdout[,norm_vars])
NB_result_train <- data.frame(dir=d_train[,'dir'],pred=NB_pred_train)
NB_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=NB_pred_holdout)
accuracy_train <- sum(diag(table(NB_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(NB_result_holdout))/dim(d_holdout)[1])
NB_performance <- data.frame(accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout)
#table
compare_list_norm <- list(ANN=ann_performance,SVM=svm_performance,RandomForest=rf_performance,NB=NB_performance)
#PARTII: discretised data
#choose ANN parameters
ann_ps_pred_train <- data.frame(n=factor(),accuracy_train=factor())
ann_ps_pred_holdout <- data.frame(n=factor(),accuracy_holdout=factor())
for (n in seq(10,100,10)){
  ann_model <- neuralnet(dir=='1' ~., data=d_ps_train[,c('dir',dc_vars)], hidden=n, learningrate=0.1,linear.output = FALSE,stepmax=1e+06)
  ann_pred_train <- ifelse(predict(ann_model,newdata = d_ps_train[,dc_vars])>=0.5,1,0)
  ann_pred_holdout <- ifelse(predict(ann_model,newdata = d_ps_holdout[,dc_vars])>=0.5,1,0)
  ann_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=ann_pred_train)
  ann_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=ann_pred_holdout)
  ann_ps_pred_train <- rbind(ann_ps_pred_train,data.frame(n=n,accuracy_train=sum(diag(table(ann_result_train))/dim(d_ps_train)[1])))
  ann_ps_pred_holdout <- rbind(ann_ps_pred_holdout,data.frame(n=n,accuracy_holdout=sum(diag(table(ann_result_holdout))/dim(d_ps_holdout)[1])))
  ann_ps_pred <- merge(ann_ps_pred_train,ann_ps_pred_holdout,by='n')
  ann_ps_pred$mean <- (ann_ps_pred$accuracy_train+ann_ps_pred$accuracy_holdout)/2
}
ann_parameter <- ann_ps_pred[order(ann_ps_pred$mean,decreasing = TRUE),][1:3,1]
#choose SVM parameters
svm_ps_pred_train <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_train=factor())
svm_ps_pred_holdout <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_holdout=factor())
for (kernel in c('polydot','rbfdot')){
  if(kernel=='polydot'){
    for (degree in 1:4){
      for (c in c(0.5,1,5,10,100)){
        svm_model <- ksvm(dir~., data=d_ps_train[,c('dir',dc_vars)], kernel=kernel, degree=degree, C=c)
        svm_pred_train <- predict(svm_model,newdata = d_ps_train[,c('dir',dc_vars)])
        svm_pred_holdout <- predict(svm_model,newdata = d_ps_holdout[,c('dir',dc_vars)])
        svm_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=svm_pred_train)
        svm_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=svm_pred_holdout)
        svm_ps_pred_train <- rbind(svm_ps_pred_train,data.frame(kernel=kernel,degree=degree,gamma=NA,c=c,accuracy_train=sum(diag(table(svm_result_train))/dim(d_ps_train)[1])))
        svm_ps_pred_holdout <- rbind(svm_ps_pred_holdout,data.frame(kernel=kernel,degree=degree,gamma=NA,c=c,accuracy_holdout=sum(diag(table(svm_result_holdout))/dim(d_ps_holdout)[1])))
      }
    }
  } else if(kernel=='rbfdot'){
    for (gamma in c(seq(0.5,5,0.5),10)){
      for (c in c(0.5,1,5,10)){
        svm_model <- ksvm(dir~., data=d_ps_train[,c('dir',dc_vars)], kernel=kernel, sigma=gamma, C=c)
        svm_pred_train <- predict(svm_model,newdata = d_ps_train[,c('dir',dc_vars)])
        svm_pred_holdout <- predict(svm_model,newdata = d_ps_holdout[,c('dir',dc_vars)])
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
#choose randomForest parameters
rf_ps_pred_train <- data.frame(ntree=factor(),accuracy_train=factor())
rf_ps_pred_holdout <- data.frame(ntree=factor(),accuracy_holdout=factor())
for (i in seq(10,200,10)){
  rf_model <- randomForest(dir~., data=d_ps_train[,c('dir',dc_vars)], ntree=i,importance = T)
  rf_pred_train <- predict(rf_model,newdata = d_ps_train[,c('dir',dc_vars)])
  rf_pred_holdout <- predict(rf_model,newdata = d_ps_holdout[,c('dir',dc_vars)])
  rf_result_train <- data.frame(dir=d_ps_train[,'dir'],pred=rf_pred_train)
  rf_result_holdout <- data.frame(dir=d_ps_holdout[,'dir'],pred=rf_pred_holdout)
  rf_ps_pred_train <- rbind(rf_ps_pred_train,data.frame(ntree=i,accuracy_train=sum(diag(table(rf_result_train))/dim(d_ps_train)[1])))
  rf_ps_pred_holdout <- rbind(rf_ps_pred_holdout,data.frame(ntree=i,accuracy_holdout=sum(diag(table(rf_result_holdout))/dim(d_ps_holdout)[1])))
  rf_ps_pred <- merge(rf_ps_pred_train,rf_ps_pred_holdout,by='ntree')
  rf_ps_pred$mean <- (rf_ps_pred$accuracy_train+rf_ps_pred$accuracy_holdout)/2
}
rf_parameter <- rf_ps_pred[order(rf_ps_pred$mean,decreasing = TRUE),][1:3,1]
#ANN
ann_performance <- data.frame(n=factor(),accuracy_train=factor(),accuracy_holdout=factor())
for (n in ann_parameter){
  ann_model <- neuralnet(dir=='1'~., data=d_train[,c('dir',dc_vars)], hidden=n,learningrate = 0.1,linear.output = FALSE,threshold = 0.01,stepmax = 1e+06)
  ann_pred_train <- ifelse(predict(ann_model,newdata = d_train[,dc_vars])>=0.5,1,0)
  ann_pred_holdout <- ifelse(predict(ann_model,newdata = d_holdout[,dc_vars])>=0.5,1,0)
  ann_result_train <- data.frame(dir=d_train[,'dir'],pred=ann_pred_train)
  ann_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=ann_pred_holdout)
  accuracy_train <- sum(diag(table(ann_result_train))/dim(d_train)[1])
  accuracy_holdout <- sum(diag(table(ann_result_holdout))/dim(d_holdout)[1])
  ann_performance <- rbind(ann_performance,data.frame(n=n,accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
}
ann_performance$mean <- (as.numeric(ann_performance$accuracy_train)+as.numeric(ann_performance$accuracy_holdout))/2
#SVM
svm_performance <- data.frame(kernel=factor(),degree=factor(),gamma=factor(),c=factor(),accuracy_train=factor(),accuracy_holdout=factor())
svm_model <- ksvm(dir~., data=d_train[,c('dir',dc_vars)], kernel=as.character(svm_parameter[1,1]), degree=as.numeric(svm_parameter[1,2]), C=as.numeric(svm_parameter[1,4]))
svm_pred_train <- predict(svm_model,newdata = d_train[,c('dir',dc_vars)])
svm_pred_holdout <- predict(svm_model,newdata = d_holdout[,c('dir',dc_vars)])
svm_result_train <- data.frame(dir=d_train[,'dir'],pred=svm_pred_train)
svm_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=svm_pred_holdout)
accuracy_train <- sum(diag(table(svm_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(svm_result_holdout))/dim(d_holdout)[1])
svm_performance <- rbind(svm_performance,data.frame(kernel=svm_parameter[1,1],degree=svm_parameter[1,2],gamma=NA,c=svm_parameter[1,4],accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
svm_model <- ksvm(dir~., data=d_train[,c('dir',dc_vars)], kernel=as.character(svm_parameter[2,1]), sigma=as.numeric(svm_parameter[2,3]), C=as.numeric(svm_parameter[2,4]))
svm_pred_train <- predict(svm_model,newdata = d_train[,c('dir',dc_vars)])
svm_pred_holdout <- predict(svm_model,newdata = d_holdout[,c('dir',dc_vars)])
svm_result_train <- data.frame(dir=d_train[,'dir'],pred=svm_pred_train)
svm_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=svm_pred_holdout)
accuracy_train <- sum(diag(table(svm_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(svm_result_holdout))/dim(d_holdout)[1])
svm_performance <- rbind(svm_performance,data.frame(kernel=svm_parameter[2,1],degree=NA,gamma=svm_parameter[2,3],c=svm_parameter[2,4],accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
svm_performance$mean <- (as.numeric(svm_performance$accuracy_train)+as.numeric(svm_performance$accuracy_holdout))/2
#randomForest
rf_performance <- data.frame(ntree=factor(),accuracy_train=factor(),accuracy_holdout=factor())
for (n in rf_parameter){
  rf_model <- randomForest(dir~., data=d_train[,c('dir',dc_vars)], ntree=i,importance = T)
  rf_pred_train <- predict(rf_model,newdata = d_train[,c('dir',dc_vars)])
  rf_pred_holdout <- predict(rf_model,newdata = d_holdout[,c('dir',dc_vars)])
  rf_result_train <- data.frame(dir=d_train[,'dir'],pred=rf_pred_train)
  rf_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=rf_pred_holdout)
  accuracy_train <- sum(diag(table(rf_result_train))/dim(d_train)[1])
  accuracy_holdout <- sum(diag(table(rf_result_holdout))/dim(d_holdout)[1])
  rf_performance <- rbind(rf_performance,data.frame(ntree=n,accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout))
}
rf_performance$mean <- (as.numeric(rf_performance$accuracy_train)+as.numeric(rf_performance$accuracy_holdout))/2
#naive Bayes
NB_model <- naive_bayes(dir ~ ., data = d_train[,c('dir',dc_vars)])
NB_pred_train <- predict(NB_model, newdata = d_train[,dc_vars])
NB_pred_holdout <- predict(NB_model, newdata = d_holdout[,dc_vars])
NB_result_train <- data.frame(dir=d_train[,'dir'],pred=NB_pred_train)
NB_result_holdout <- data.frame(dir=d_holdout[,'dir'],pred=NB_pred_holdout)
accuracy_train <- sum(diag(table(NB_result_train))/dim(d_train)[1])
accuracy_holdout <- sum(diag(table(NB_result_holdout))/dim(d_holdout)[1])
NB_performance <- data.frame(accuracy_train=accuracy_train,accuracy_holdout=accuracy_holdout)
#table
compare_list_dc <- list(ANN=ann_performance,SVM=svm_performance,RandomForest=rf_performance,NB=NB_performance)