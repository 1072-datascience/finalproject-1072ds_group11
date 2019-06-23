##record what I want to do in this shiny app

shinyApp(ui = ui, server = server)

#publish
library(rsconnect)
deployApp("C:/Users/User/Documents/ActivityRecognition/advanced")