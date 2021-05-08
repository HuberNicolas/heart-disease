suppressPackageStartupMessages(library(tensorflow))
suppressPackageStartupMessages(library(caTools))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(keras))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(plotly))

# X and y represent the filtered (25 features) data after doing Random Forests
X <- read.csv('X.csv')
y <- read.csv('y.csv')

X <- select(X, -X)
y <- select(y, -X)

head(X)

###
#set.seed(101)
#sampleX <- sample.int(282, 211, replace = FALSE)

#X_train <- X[sampleX,]
#y_train <- y[sampleX,]

#X_test <- X[-sampleX,]
#y_test <- y[-sampleX,]
###

minmax <- function(x) (x - min(x))/(max(x) - min(x))

X <- apply(X, 2, minmax)

###
#X <- as.data.frame(X)
#X$target <- y
###

###
#X_train <- apply(X_train, 2, minmax)
#X_test <- apply(X_test, 2, minmax)
#x_train <- as.matrix(X_train)
###

X <- as.matrix(X)

# Build the autoencoders with a bottleneck of 2 dimensions
model <- keras_model_sequential()
model %>%
  layer_dense(units = 6, activation = "relu", input_shape = ncol(X)) %>%
  layer_dense(units = 2, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 6, activation = "sigmoid") %>%
  layer_dense(units = ncol(X))
summary(model)

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = "adam"
)

# Fit the model
model %>% fit(
  x = X, 
  y = X, 
  epochs = 50,
  batch_size = 16,
)

# Evaluate the mode
eval <- evaluate(model, X, X)

# Extract the information from the bottleneck
bottleneck_model <- keras_model(inputs = model$input, outputs = get_layer(model, "bottleneck")$output)
bottleneck_model_output <- predict(bottleneck_model, X)

# Plot the dataset in a lower dimension
ggplot(data.frame(PC1 = bottleneck_model_output[,1], PC2 = bottleneck_model_output[,2]), 
       aes(x = PC1, y = PC2, col = y$num)) + 
  geom_point() +
  xlab("") + ylab("") +
  labs(color = "Target") +
  ggtitle("Dimension reduction using Autoencoders") 


# In 3D:
model3D <- keras_model_sequential()
model3D %>%
  layer_dense(units = 6, activation = "relu", input_shape = ncol(X)) %>%
  layer_dense(units = 3, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 6, activation = "sigmoid") %>%
  layer_dense(units = ncol(X))
summary(model3D)


# compile model
model3D %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = "adam"
)

# fit model
model3D %>% fit(
  x = X, 
  y = X, 
  epochs = 50,
  batch_size = 16,
  verbose = 0
)

# evaluate the model
eval3D <- evaluate(model3D, X, X)

# Extract the information from the bottleneck
bottleneck_model3D <- keras_model(inputs = model3D$input, outputs = get_layer(model3D, "bottleneck")$output)
bottleneck_model_output3D <- predict(bottleneck_model3D, X)

# Plot the dataset in a lower dimension
df3D <- data.frame(layer1 = bottleneck_model_output3D[,1], 
                   layer2 = bottleneck_model_output3D[,2], 
                   layer3 = bottleneck_model_output3D[,3])
plot_ly(df3D, x = ~layer1, y = ~layer2, z = ~layer3, color = ~y$num) %>% 
  add_markers()













