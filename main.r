mydata <- read.csv("/Users/nick/PyProjects/WatermelonBook/data/watermelon_3a.csv", header = FALSE)
head(mydata)
X = mydata[, 2:3]
y = mydata[, 4]
lgr = glm(formula = y ~ ., family = binomial(link = "logit"), data = X)
summary(lgr)
