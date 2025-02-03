library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(laGP)
library(dplyr)
library(ggplot2)
library(stats)

n_tr <- 10000 # number of training data
n_ts <- 5000 # number of test data
val_fac <- 0.8 # train/validation separate factor
f <- 16 # number of features (X dimension)
nn_tr <- 100 # number of new class data, use maximum it can be if exceeds the max
d <- 10 # GP lengthscale

tau <- 0.90 # threshold used in objective function
c = 4 # GP class to update
l = 7 # new class to add to GPc
gamma_cl = 4.68 # optimal mean used for class l to retrain GP_{cl} (yy)

directory_path <- paste0("Rdata_ntr", toString(n_tr), "_f", toString(f))
pdf(file = "Rplots_Option2.pdf")
if (!dir.exists(directory_path)) {
  if (dir.create(directory_path)) {
    cat("Directory created successfully:", directory_path, "\n")
  } else {
    cat("Failed to create the directory:", directory_path, "\n")
  }
} else {
  cat("Directory already exists:", directory_path, "\n")
}

# Store and access the data as dictionary, key is "c0, c1, ..., c9" for each class respectively
all_data_X <- list()
all_data_Y <- list()
val_data_X <- list()
val_data_Y <- list()
test_data_X <- list()
test_data_Y <- list()
test_data_label <- list() # for convenient

GPmodel_train <- list()
GPresult_train <- list()
mse_train <- list()

######### Train GP0 - GP4 #########
df <- read.csv(paste0("out/f_", toString(f), "/train.csv"))
set.seed(430)
select.index <- sample(1:nrow(df), n_tr, replace = FALSE)
train.df <- df[select.index, ]
n_train <- floor(n_tr * val_fac)
val.df <- train.df[(n_train + 1):n_tr, ]
train.df <- train.df[1:n_train, ]

for (j in 1:5) {
  label <- j-1
  key <- paste0("c",label)

  X <- as.matrix(train.df[train.df[, "label"] == label, 1:f])
  Y <- matrix(train.df[train.df[, "label"] == label, (f + j)])
  val.X <- as.matrix(val.df[val.df[, "label"] == label, 1:f])
  val.Y <- matrix(val.df[val.df[, "label"] == label, (f + j)])

  gpisep <- newGP(X, Y, d = d, g = 1e-4, dK = TRUE)
  out <- predGP(gpisep, X, lite = TRUE)
  
  all_data_X[[key]] <- X
  all_data_Y[[key]] <- Y
  val_data_X[[key]] <- val.X
  val_data_Y[[key]] <- val.Y
  GPresult_train[[key]] <- out
  GPmodel_train[[key]] <- gpisep

  val_result <- predGP(gpisep, val.X, lite = TRUE)
  mse <- norm(val_result$mean - val.Y, "2")
  mse_train[[key]] <- mse
  print(paste0("validation MSE for class", label, ": ", mse))
  plot(val.Y, val_result$mean, ylab = "Y_pred", xlab = "Y_true")
  title(paste0("Validation True vs. Pred (class=", label, ")"))
  rm(gpisep)
}
print("Train Finished")
print("---------------------")

######### Prepare Test Data #########
test.df = read.csv(paste0("out/f_", toString(f), "/test.csv"))
test.df <- test.df[1:n_ts, ]

for(j in 1: 10) {
  label <- j - 1
  key <- paste0("c", label)
  test_df <- test.df[test.df[, "label"] == label, ]
  test_data_X[[key]] <- as.matrix(test_df[, 1:f])
  test_data_Y[[key]] <- matrix(test_df[, (f + j)])
  test_data_label[[key]] <- matrix(label, nrow = nrow(test_data_X[[key]]), ncol=1)
}

######### Test GP0 - GP4 #########
test_oldset <- do.call(rbind, lapply(c("c0", "c1", "c2", "c3", "c4"), function(index) test_data_X[[index]]))
test_oldset_label <- do.call(rbind, lapply(c("c0", "c1", "c2", "c3", "c4"), function(index) test_data_label[[index]]))

print(paste("Test1 Data: ", nrow(test_oldset), " from classes label 0-4"))
GP_test <- vector("list", 5)
GP_test_mean <- vector("list", 5)
for (j in 1: 5) {
  key <- paste0("c", j - 1)
  out <- predGP(GPmodel_train[[key]], test_oldset, lite=TRUE)
  GP_test[[j]] <- out
  GP_test_mean[[j]] <- out$mean
}
test_result <- do.call(cbind, GP_test_mean)

correct_predictions <- sapply(1:nrow(test_result), function(i) {
  which.max(test_result[i, ]) == (test_oldset_label[i, 1] + 1)
})
test1_accuracy <- mean(correct_predictions)
print(paste0("Test1 first 5 classes test accuracy is ", test1_accuracy))

class_labels <- 0:4
accuracy_per_label <- sapply(class_labels, function(label) {
  label_indices <- which(test_oldset_label[, 1] == label)
  print(paste0("Class Label ", label, " TPR is ", mean(correct_predictions[label_indices])))
})
print("---------------------")


######### Add New Class (label 5-9) #########
NNscores_sd <- vector("list", 5)
for(j in 1:5){
    label <- j + 4
    key <- paste0("c", label)
    num_rows <- nrow(train.df[train.df[, "label"] == label, ])
    nn_tr <- min(nn_tr, num_rows)
    print(paste0("New class ", label, " has ", nn_tr, " rows of data"))
    indices <- sample(num_rows, nn_tr, replace = FALSE)
    select_newClass <- as.matrix(train.df[train.df[, "label"] == label, ])
    select_newClass <- select_newClass[indices, ]
    all_data_X[[key]] <- select_newClass[, 1:f]
    all_data_Y[[key]] <- select_newClass[, (f + label + 1), drop = FALSE]
    
    select_newClass_val <- as.matrix(val.df[val.df[, "label"] == label, ])
    val_data_X[[key]] <- select_newClass_val[, 1:f]
    val_data_Y[[key]] <- select_newClass_val[, (f + label + 1), drop = FALSE]

    # calculate sd from nn
    NNscores_sd[[key]] <- sd(all_data_Y[[key]])
    #print(NNscores_sd[[key]])
}

######### GPc orginal  + class l with NN sd#########
OPTIMAL_GP <- paste0("c", c)
OPTIMAL_l <- paste0("c", l)

existingclass_set <- list(0, 1, 2, 3, 4)
GP_preds_mean <- lapply(existingclass_set, function(i) {
      predGP(GPmodel_train[[OPTIMAL_GP]], all_data_X[[paste0("c",i)]], lite = TRUE)$mean
})
GP_preds_mu <- sapply(GP_preds_mean, mean)
GP_preds_sd <- sapply(GP_preds_mean, sd)
# plot
x <- seq(-10, 15, length.out = 500)
    density_data <- c(
      sapply(seq_along(GP_preds_mu), function(i) dnorm(x, mean = GP_preds_mu[i], sd = GP_preds_sd[i])),
      dnorm(x, mean = gamma_cl, sd = NNscores_sd[[OPTIMAL_l]]))
data <- data.frame(
  x = rep(x, length(density_data)/ length(x)),
  density = density_data,
  group = factor(rep(c(paste0("class ", existingclass_set), paste0("class ", l, " with mean=", gamma_cl)), each = length(x))))
plot <- ggplot(data, aes(x = x, y = density, color = group)) +
  geom_line(size = 1) +
  labs(title = paste0("GP", OPTIMAL_GP, " output distribution on different classes"), x = "y-score", y = "Density") +
  theme_minimal() + 
  ylim(0, 0.1)
print(plot)


######### Update GPc #########
X_on <- as.matrix(all_data_X[[OPTIMAL_GP]])
label_on <- matrix(0, nrow = nrow(X_on), ncol = 1) # for logistic classifer, 0 = old class
X_on <- rbind(X_on, all_data_X[[OPTIMAL_l]])
label_on <- rbind(label_on, matrix(1, nrow = nrow(all_data_X[[OPTIMAL_l]]), ncol = 1))  # for logistic classifer, 1 = new class

Y_on <- all_data_Y[[OPTIMAL_GP]]

# OPTION1: new class l, Fix y = gamma_cl
#Y_on <- rbind(Y_on, matrix(gamma_cl, nrow = nrow(all_data_X[[OPTIMAL_l]]), ncol = 1))

# OPTION2: new class l, shift NN distribution to distribution with mean=gamma_cl
Y_on <- rbind(Y_on, matrix(gamma_cl + all_data_Y[[OPTIMAL_l]] - mean(all_data_Y[[OPTIMAL_l]]), nrow = nrow(all_data_X[[OPTIMAL_l]]), ncol = 1))

gpisep <- newGP(X_on, Y_on, d = d, g = 1e-4, dK = TRUE)
out <- predGP(gpisep, X_on, lite = TRUE)
GPresult_train[[OPTIMAL_GP]] <- out
GPmodel_train[[OPTIMAL_GP]] <- gpisep
print("Retrain Finished")
print("---------------------")

GP_preds_mean <- lapply(existingclass_set, function(i) { # list of 5, each of length = number of traning sample in class i
  predGP(GPmodel_train[[OPTIMAL_GP]], all_data_X[[paste0("c",i)]], lite = TRUE)$mean
})
GP_preds_mu <- sapply(GP_preds_mean, mean)
GP_preds_sd <- sapply(GP_preds_mean, sd)

print("new distribution after retrain for existing classes")
print(GP_preds_mu)
print(GP_preds_sd)

l_pred <- predGP(GPmodel_train[[OPTIMAL_GP]], all_data_X[[OPTIMAL_l]], lite = TRUE)$mean
l_mu <- mean(l_pred)
l_sd <- sd(l_pred)
print(paste0("Retrained, new class l mean=", l_mu,", sd=", l_sd))

# plot
x <- seq(-10, 15, length.out = 500)
density_data <- c(
  sapply(seq_along(GP_preds_mu), function(i) dnorm(x, mean = GP_preds_mu[i], sd = GP_preds_sd[i])),
  dnorm(x, mean = l_mu, sd = l_sd)
)
data <- data.frame(
  x = rep(x, length(density_data)/ length(x)),
  density = density_data,
  group = factor(rep(c(paste0("class ", existingclass_set), paste0("class ", l, " with mean=", gamma_cl)), each = length(x)))
)
plot <- ggplot(data, aes(x = x, y = density, color = group)) +
  geom_line(size = 1) +
  labs(title = paste0("Retrained GP", OPTIMAL_GP, " output distribution on different classes"), x = "y-score", y = "Density") +
  theme_minimal() + 
  ylim(0, 0.1)
print(plot)

# plot histogram
preds_df <- do.call(rbind, lapply(seq_along(GP_preds_mean), function(i) {
  data.frame(
    score = GP_preds_mean[[i]],
    class = paste0("class ", existingclass_set[i])
  )
}))
preds_df <- rbind(preds_df, data.frame(
  score = l_pred,
  class = paste0("class ", l)
))
histogram <- ggplot(preds_df, aes(x = score, fill = class)) +
  geom_histogram(binwidth = 0.5, alpha = 0.7, position = "identity") +
  labs(title = paste0("Retrained GP", OPTIMAL_GP, " histogram of predGP scores by class"), x = "y-score", y = "Count") +
  theme_minimal() + 
  ylim(0, 1000)
print(histogram)

## Testset ##
l_pred_test <- predGP(GPmodel_train[[OPTIMAL_GP]], test_data_X[[OPTIMAL_l]], lite = TRUE)$mean
l_mu_test <- mean(l_pred_test)
l_sd_test <- sd(l_pred_test)
print(paste0("Retrained, new class l testset mean=", l_mu_test,", sd=", l_sd_test))

other_GP_preds_mean <- lapply(existingclass_set, function(i) {
  predGP(GPmodel_train[[OPTIMAL_GP]], test_data_X[[paste0("c",i)]], lite = TRUE)$mean
})
other_GP_preds_mu <- sapply(other_GP_preds_mean, mean)
other_GP_preds_sd <- sapply(other_GP_preds_mean, sd)

print("new distribution after retrain for existing classes, testset")
print(other_GP_preds_mu)
print(other_GP_preds_sd)

# plot
x <- seq(-10, 15, length.out = 500)
density_data <- c(
  sapply(seq_along(other_GP_preds_mu), function(i) dnorm(x, mean = other_GP_preds_mu[i], sd = other_GP_preds_sd[i])),
  dnorm(x, mean = l_mu_test, sd = l_sd_test)
)
data <- data.frame(
  x = rep(x, length(density_data) / length(x)),
  density = density_data,
  group = factor(rep(c(paste0("class ", existingclass_set), paste0("class ", l)), each = length(x)))
)
plot <- ggplot(data, aes(x = x, y = density, color = group)) +
  geom_line(size = 1) +
  labs(title = paste0(" Retrained GP", OPTIMAL_GP, " output distribution on different classes (testset)"), x = "y-score", y = "Density") +
  theme_minimal() + 
  ylim(0, 0.1)
print(plot)

# plot histogram
preds_df <- do.call(rbind, lapply(seq_along(other_GP_preds_mean), function(i) {
  data.frame(
    score = other_GP_preds_mean[[i]],
    class = paste0("class ", existingclass_set[i])
  )
}))
preds_df <- rbind(preds_df, data.frame(
  score = l_pred_test,
  class = paste0("class ", l)
))
histogram <- ggplot(preds_df, aes(x = score, fill = class)) +
  geom_histogram(binwidth = 0.5, alpha = 0.7, position = "identity") +
  labs(title = paste0("Retrained GP", OPTIMAL_GP, " histogram of predGP acores by class (testset)"), x = "y-score", y = "Count") +
  theme_minimal() + 
  ylim(0, 1000)
print(histogram)

######### Train a logistic classifier for classes c and l #########
y_on_GPon <- data.frame(predGP(GPmodel_train[[OPTIMAL_GP]], X_on, lite = TRUE)$mean)
y_on_GPon$label <- label_on
colnames(y_on_GPon)[1] <- "value"
logistic_model <- glm(label ~ ., data = y_on_GPon, family = binomial)
print("Classifier Train Finished")
print("---------------------")

######### Test After Retrained GP_{cl} #########
test_X <- do.call(rbind, lapply(c("c0", "c1", "c2", "c3", "c4", paste0("c", l)), function(index) test_data_X[[index]]))
test_X_label <- do.call(rbind, lapply(c("c0", "c1", "c2", "c3", "c4", paste0("c", l)), function(index) test_data_label[[index]]))

test_preds <- sapply(existingclass_set, function(i) {
  predGP(GPmodel_train[[paste0("c", i)]], test_X, lite = TRUE)$mean
})

predicted_class_indices <- max.col(test_preds, ties.method = "first")
predicted_classes <- sapply(predicted_class_indices, function(index) {
  existingclass_set[index]
})

class_c_indices <- which(predicted_classes  == c)
if(length(class_c_indices) > 0){
  X_c <- test_X[class_c_indices, ]
  prob_class_l <- predict(logistic_model, newdata = data.frame(value = predGP(GPmodel_train[[OPTIMAL_GP]], X_c, lite = TRUE)$mean))
  final_class_cl <- ifelse(prob_class_l > 0.5, l, c)
  predicted_classes[class_c_indices] <- final_class_cl
}

correct_predictions <- sapply(1:nrow(test_X), function(i) {
  predicted_classes[i] == test_X_label[i, 1]
})
test2_accuracy <- mean(correct_predictions)

print(paste0("Test2 first 5 classes + class ", l, "test accuracy is ", test2_accuracy))
class_labels <- c(0:4, l)
accuracy_per_label <- sapply(class_labels, function(label) {
  label_indices <- which(test_X_label[, 1] == label)
  print(paste0("Class Label ", label, " TPR is ", mean(correct_predictions[label_indices])))
})

######### Save Results #########
while (!is.null(dev.list()))  dev.off()