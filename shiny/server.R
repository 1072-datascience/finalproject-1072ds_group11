library(shiny)
library(ggplot2)
library(ggbiplot)
server = function(input, output){
  
  output$origin_plot = renderPlot({
    load(paste0(input$stock, "1.rda"))
    i = switch(input$parameter,
               SMA = 1,
               WMA = 2,
               MOMENTUM = 3,
               K = 4,
               D = 5,
               RSI = 6,
               MACD = 7,
               R = 8,
               ADX = 9,
               CCI = 10
    )
    print(ggplot(data = d, aes(x = c(1:nrow(d)), y = d[[i + 6]], color = d$dir)) +
            geom_line() + ylab(names(d)[[i + 6]]) + xlab("Date"))
  })
  
  output$origin_pca = renderPlot({
    load(paste0(input$stock, "1.rda"))
    x = switch(input$pc1,
               PC1 = 1,
               PC2 = 2,
               PC3 = 3,
               PC4 = 4,
               PC5 = 5,
               PC6 = 6,
               PC7 = 7,
               PC8 = 8,
               PC9 = 9,
               PC10 = 10)
    y = switch(input$pc1,
               PC1 = 1,
               PC2 = 2,
               PC3 = 3,
               PC4 = 4,
               PC5 = 5,
               PC6 = 6,
               PC7 = 7,
               PC8 = 8,
               PC9 = 9,
               PC10 = 10)
    d_na = na.omit(d)
    d_pca = prcomp(d_na[7:16], center = TRUE, scale. = TRUE)
    ggbiplot(d_pca, obs.scale = 1, var.scale = 1, choices = c(x, y), groups = d_na$dir, varname.size = 0, circle = TRUE, ellipse = TRUE)
  })
  
  output$scale_plot = renderPlot({
    load(paste0(input$stock, "1.rda"))
    i = switch(input$parameter,
               SMA = 1,
               WMA = 2,
               MOMENTUM = 3,
               K = 4,
               D = 5,
               RSI = 6,
               MACD = 7,
               R = 8,
               ADX = 9,
               CCI = 10
    )
    print(ggplot(data = d, aes(x = c(1:nrow(d)), y = d[[i + 20]], color = d$dir)) +
            geom_line() + ylab(names(d)[[i + 20]]) + xlab("Date"))
  })
  
  output$scale_pca = renderPlot({
    load(paste0(input$stock, "1.rda"))
    x = switch(input$pc1,
               PC1 = 1,
               PC2 = 2,
               PC3 = 3,
               PC4 = 4,
               PC5 = 5,
               PC6 = 6,
               PC7 = 7,
               PC8 = 8,
               PC9 = 9,
               PC10 = 10)
    y = switch(input$pc1,
               PC1 = 1,
               PC2 = 2,
               PC3 = 3,
               PC4 = 4,
               PC5 = 5,
               PC6 = 6,
               PC7 = 7,
               PC8 = 8,
               PC9 = 9,
               PC10 = 10)
    d_na = na.omit(d)
    d_pca = prcomp(d_na[21:30], center = TRUE, scale. = TRUE)
    ggbiplot(d_pca, obs.scale = 1, var.scale = 1, choices = c(x, y), groups = d_na$dir, varname.size = 0, circle = TRUE, ellipse = TRUE)
  })
  
  output$discrete_plot = renderPlot({
    load(paste0(input$stock, "1.rda"))
    i = switch(input$parameter,
               SMA = 1,
               WMA = 2,
               MOMENTUM = 9,
               K = 3,
               D = 4,
               RSI = 7,
               MACD = 6,
               R = 5,
               CCI = 8)
    print(ggplot(data = d, aes(d[[i + 30]], ..count..)) + geom_bar(aes(fill = d$dir)) + xlab(names(d)[[i +30]]))
    
  })
}
