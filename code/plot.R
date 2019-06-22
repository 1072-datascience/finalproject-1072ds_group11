### packages
library(caret)
library(gridExtra)
### in this R code, we will produce the plots that need to show in presentation.

### index increasing or decreasing per year
load("Japan.rda")

for(i in c(7:16, 21:30)){
  png(filename = paste0("png/", names(d)[[i]], ".png"))
  print(ggplot(data = d, aes(x = c(1:nrow(d)), y = d[[i]], color = d$dir)) +
    geom_line() + ylab(names(d)[[i]]) + xlab("Date"))
  dev.off()
}
for(i in 31:39){
  png(filename = paste0("png/", names(d)[[i]], ".png"))
  print(ggplot(data = d, aes(d[[i]], ..count..)) + geom_bar(aes(fill = d$dir)) + xlab(names(d)[[i]]))
  dev.off()
}

###