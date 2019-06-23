ui = fluidPage(
  #App title
  titlePanel("Stock Index"),
  
  #Sidebar layout
  sidebarLayout(
    #Input
    sidebarPanel(
      selectInput("stock", "Stock", c("JPY", "TWD", "GBP"), multiple = F),
      
      br(),
      
      selectInput("parameter", "Financial Parameter", c("SMA", "WMA", "MOMENTUM", "K", "D", "RSI", "MACD", "R", "ADX", "CCI")),
      
      br(),
      
      selectInput("pc1", "Principle Component X", c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10")),
      
      br(),
      
      selectInput("pc2", "Principle Component Y", c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"), selected = "PC2")
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
