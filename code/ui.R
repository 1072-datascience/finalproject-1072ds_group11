ui = fluidPage(
  #App title
  titlePanel("Stock Index"),
  
  #Sidebar layout
  sidebarLayout(
    #Input
    sidebarPanel(
      selectInput("stock", "Stock", c("JPY", "TWD", "GBP"), multiple = F),
      
      br(),
      
      selectInput("parameter", "Financial Parameter", c("SMA", "WMA", "MOMENTUM", "K", "D", "RSI", "MACD", "R", "ADX", "CCI"))
    ),
    
    mainPanel(
      tabsetPanel(type = "tabs",
                  tabPanel("Origin", plotOutput("origin_plot"), plotOutput("origin_pca")),
                  tabPanel("Scale", plotOutput("scale_plot"), plotOutput("scale_pca")),
                  tabPanel("Discrete", plotOutput("discrete_plot"))
      )
    )
  )
)
