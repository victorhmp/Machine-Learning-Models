# Decision Tree Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting the Regression model to the dataset
# install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))


# Visualising the Decision Tree Regression Model results (for better resolution and smoother curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             color = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), 
            color = 'blue') +
  ggtitle('Thruth or Bluff (Decision Tree Regression Model)') +
  xlab('Level') +
  ylab('Salary')
