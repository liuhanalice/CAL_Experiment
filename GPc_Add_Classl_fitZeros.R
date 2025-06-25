# When training a GP model, also include other classes' data (whose y score is around zero) to train the model
library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(laGP)
library(dplyr)
library(ggplot2)
library(stats)

n_tr <- 10000 # number of training data
n_ts <- 4000 # number of test data
val_fac <- 0.8 # train/validation separate factor
f <- 16 # number of features (X dimension)
nn_tr <- 400 # number of new class data, use maximum it can be if exceeds the max
d <- 10 # GP lengthscale
other_class_sample_num <- 200 # total number of other class data (whose y socre is around zero) to include in training

tau <- 0.90 # threshold used in objective function
c = 4 # GP class to update
l = 7 # new class to add to GPc
gamma_cl = 4.68 # optimal mean used for class l to retrain GP_{cl} (yy)

directory_path <- paste0("Rdata_ntr", toString(n_tr), "_f", toString(f))
pdf(file = "Rplots_Option2_filteredTraining_fitZeros.pdf")
if (!dir.exists(directory_path)) {
  if (dir.create(directory_path)) {
    cat("Directory created successfully:", directory_path, "\n")
  } else {
    cat("Failed to create the directory:", directory_path, "\n")
  }
} else {
  cat("Directory already exists:", directory_path, "\n")
}

# Functions
random_select_rows <- function(mat, num_rows) {
  if (!is.matrix(mat)) stop("Input must be a matrix")
  if (num_rows > nrow(mat)) stop("num_rows cannot be greater than the number of rows in the matrix")
  
  selected_rows <- sample(nrow(mat), num_rows)
  return(mat[selected_rows, , drop = FALSE])
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
df <- read.csv(paste0("out/f_", toString(f), "/filtered_train.csv"))
set.seed(430)
select.index <- sample(1:nrow(df), n_tr, replace = FALSE)
train.df <- df[select.index, ]
n_train <- floor(n_tr * val_fac)
val.df <- train.df[(n_train + 1):n_tr, ]
train.df <- train.df[1:n_train, ]

existingclass_set <- list(0, 1, 2, 3, 4)
for (j in 1:5) {
  label <- j-1
  key <- paste0("c",label)

  other_class <- as.matrix(train.df[train.df[, "label"] %in% setdiff(existingclass_set, label), ])
  sampled_other <- random_select_rows(other_class, other_class_sample_num)
  other_class_val <- as.matrix(val.df[val.df[, "label"] %in% setdiff(existingclass_set, label), ])
  sampled_other_val <- random_select_rows(other_class_val, other_class_sample_num)

  X <- as.matrix(train.df[train.df[, "label"] == label, 1:f])
  X <- rbind(X, sampled_other[, 1:f])
  Y <- matrix(train.df[train.df[, "label"] == label, (f + j)])
  Y <- rbind(Y, sampled_other[, (f + j), drop = FALSE])

  val.X <- as.matrix(val.df[val.df[, "label"] == label, 1:f])
  val.X <- rbind(val.X, sampled_other_val[, 1:f])
  val.Y <- matrix(val.df[val.df[, "label"] == label, (f + j)])
  val.Y <- rbind(val.Y, sampled_other_val[, (f + j), drop = FALSE])

  gpisep <- newGPsep(X, Y, d = rep(d, ncol(X)), g = 1e-4, dK = TRUE)
  mleGPsep(gpisep, param="d", tmin=rep(0.5,ncol(X)), tmax=rep(50,ncol(X)))
  out <- predGPsep(gpisep, X, lite = TRUE)
  
  all_data_X[[key]] <- as.matrix(train.df[train.df[, "label"] == label, 1:f])
  all_data_Y[[key]] <- matrix(train.df[train.df[, "label"] == label, (f + j)])
  val_data_X[[key]] <- as.matrix(val.df[val.df[, "label"] == label, 1:f])
  val_data_Y[[key]] <- matrix(val.df[val.df[, "label"] == label, (f + j)])
  GPresult_train[[key]] <- out
  GPmodel_train[[key]] <- gpisep

  val_result <- predGPsep(gpisep, val.X, lite = TRUE)
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
  out <- predGPsep(GPmodel_train[[key]], test_oldset, lite=TRUE)
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

# plot [testset]
OPTIMAL_GP <- paste0("c", c)
OPTIMAL_l <- paste0("c", l)

existingclass_set <- list(0, 1, 2, 3, 4)
GP_preds_mean <- lapply(existingclass_set, function(i) {
      predGPsep(GPmodel_train[[OPTIMAL_GP]], test_data_X[[paste0("c",i)]], lite = TRUE)$mean
})
GP_preds_mu <- sapply(GP_preds_mean, mean)
GP_preds_sd <- sapply(GP_preds_mean, sd)

x <- seq(-10, 15, length.out = 500)
density_data <- c(
    sapply(seq_along(GP_preds_mu), function(i) dnorm(x, mean = GP_preds_mu[i], sd = GP_preds_sd[i])),
    dnorm(x, mean = 7.3253690846, sd = 0.4514986690138651))
data <- data.frame(
  x = rep(x, length(density_data)/ length(x)),
  density = density_data,
  group = factor(rep(c(paste0("class ", existingclass_set), "class 4 NN testset distribution"), each = length(x))))
plot <- ggplot(data, aes(x = x, y = density, color = group)) +
  geom_line(size = 1) +
  labs(title = paste0("GP", OPTIMAL_GP, " output distribution on different classes (testset)"), x = "y-score", y = "Density") +
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
  score = test_data_Y[["c4"]][, 1],
  class = rep("class 4 NN", nrow(test_data_Y[["c4"]])),
  stringsAsFactors = FALSE
))
histogram <- ggplot(preds_df, aes(x = score, fill = class)) +
  geom_histogram(binwidth = 0.5, alpha = 0.7, position = "identity") +
  labs(title = paste0("GP", OPTIMAL_GP, " histogram of predGPsep scores by class (testset)"), x = "y-score", y = "Count") +
  theme_minimal() + 
  ylim(0, 1000)
print(histogram)
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
      predGPsep(GPmodel_train[[OPTIMAL_GP]], all_data_X[[paste0("c",i)]], lite = TRUE)$mean
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
# plot histogram
preds_df <- do.call(rbind, lapply(seq_along(GP_preds_mean), function(i) {
  data.frame(
    score = GP_preds_mean[[i]],
    class = paste0("class ", existingclass_set[i])
  )
}))
print(dim(all_data_Y[[OPTIMAL_l]]))
preds_df <- rbind(preds_df, data.frame(
  score = all_data_Y[[OPTIMAL_l]][, 1] + gamma_cl - mean(all_data_Y[[OPTIMAL_l]][,1]),
  class = rep(paste0("class ", l), nrow(all_data_Y[[OPTIMAL_l]])),
  stringsAsFactors = FALSE
))
histogram <- ggplot(preds_df, aes(x = score, fill = class)) +
  geom_histogram(binwidth = 0.5, alpha = 0.7, position = "identity") +
  labs(title = paste0("GP", OPTIMAL_GP, " histogram of predGPsep scores by class"), x = "y-score", y = "Count") +
  theme_minimal() + 
  ylim(0, 1000)
print(histogram)

###
range_df <- preds_df %>%
  group_by(class) %>%
  summarize(min_score = min(score), max_score = max(score))

calculate_percentage_in_range <- function(scores, min_b, max_b) {
  in_range <- scores >= min_b & scores <= max_b
  percentage_in_range <- sum(in_range) / length(scores) * 100
  return(percentage_in_range)
}

# Calculate the percentage of each group's data within the range of every other group's data
classes <- unique(preds_df$class)
percentage_matrix <- matrix(0, nrow = length(classes), ncol = length(classes))
rownames(percentage_matrix) <- classes
colnames(percentage_matrix) <- classes

for (i in 1:length(classes)) {
  for (j in 1:length(classes)) {
    if (i != j) {
      class_a <- classes[i]
      class_b <- classes[j]
      scores_a <- preds_df %>% filter(class == class_a) %>% pull(score)
      min_b <- range_df %>% filter(class == class_b) %>% pull(min_score)
      max_b <- range_df %>% filter(class == class_b) %>% pull(max_score)
      percentage_in_range <- calculate_percentage_in_range(scores_a, min_b, max_b)
      percentage_matrix[i, j] <- percentage_in_range
    }else{
      percentage_matrix[i, j] <- 100
    }
  }
}

print("Percentage matrix: before retraining [NN score shift with mean=gamma_cl]")
print(percentage_matrix)
###


######### Update GPc #########
other_class <- as.matrix(train.df[train.df[, "label"] %in% setdiff(existingclass_set, c), ])
sampled_other <- random_select_rows(other_class, other_class_sample_num)
other_class_val <- as.matrix(val.df[val.df[, "label"] %in% setdiff(existingclass_set, c), ])
sampled_other_val <- random_select_rows(other_class_val, other_class_sample_num)

X_on <- as.matrix(all_data_X[[OPTIMAL_GP]])
X_on <- rbind(X_on, sampled_other[, 1:f])
label_on <- matrix(0, nrow = nrow(X_on), ncol = 1) # for logistic classifer, 0 = old class
X_on <- rbind(X_on, all_data_X[[OPTIMAL_l]])
label_on <- rbind(label_on, matrix(1, nrow = nrow(all_data_X[[OPTIMAL_l]]), ncol = 1))  # for logistic classifer, 1 = new class

Y_on <- all_data_Y[[OPTIMAL_GP]]
Y_on <- rbind(Y_on, sampled_other[, (f + c + 1), drop = FALSE])

# OPTION1: new class l, Fix y = gamma_cl
#Y_on <- rbind(Y_on, matrix(gamma_cl, nrow = nrow(all_data_X[[OPTIMAL_l]]), ncol = 1))

# OPTION2: new class l, shift NN distribution to distribution with mean=gamma_cl
Y_on <- rbind(Y_on, matrix(gamma_cl + all_data_Y[[OPTIMAL_l]] - mean(all_data_Y[[OPTIMAL_l]]), nrow = nrow(all_data_X[[OPTIMAL_l]]), ncol = 1))

gpisep <- newGPsep(X_on, Y_on, d = rep(d, ncol(X)), g = 1e-4, dK = TRUE)
mleGPsep(gpisep, param="d", tmin=rep(0.5,ncol(X)), tmax=rep(50,ncol(X)))
out <- predGPsep(gpisep, X_on, lite = TRUE)
GPresult_train[[OPTIMAL_GP]] <- out
GPmodel_train[[OPTIMAL_GP]] <- gpisep
print("Retrain Finished")
print("---------------------")

GP_preds_mean <- lapply(existingclass_set, function(i) { # list of 5, each of length = number of traning sample in class i
  predGPsep(GPmodel_train[[OPTIMAL_GP]], all_data_X[[paste0("c",i)]], lite = TRUE)$mean
})
GP_preds_mu <- sapply(GP_preds_mean, mean)
GP_preds_sd <- sapply(GP_preds_mean, sd)

print("new distribution after retrain for existing classes")
print(GP_preds_mu)
print(GP_preds_sd)

l_pred <- predGPsep(GPmodel_train[[OPTIMAL_GP]], all_data_X[[OPTIMAL_l]], lite = TRUE)$mean
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
  labs(title = paste0("Retrained GP", OPTIMAL_GP, " histogram of predGPsep scores by class"), x = "y-score", y = "Count") +
  theme_minimal() + 
  ylim(0, 1000)
print(histogram)

###
range_df <- preds_df %>%
  group_by(class) %>%
  summarize(min_score = min(score), max_score = max(score))

calculate_percentage_in_range <- function(scores, min_b, max_b) {
  in_range <- scores >= min_b & scores <= max_b
  percentage_in_range <- sum(in_range) / length(scores) * 100
  return(percentage_in_range)
}

# Calculate the percentage of each group's data within the range of every other group's data
classes <- unique(preds_df$class)
percentage_matrix <- matrix(0, nrow = length(classes), ncol = length(classes))
rownames(percentage_matrix) <- classes
colnames(percentage_matrix) <- classes

for (i in 1:length(classes)) {
  for (j in 1:length(classes)) {
    if (i != j) {
      class_a <- classes[i]
      class_b <- classes[j]
      scores_a <- preds_df %>% filter(class == class_a) %>% pull(score)
      min_b <- range_df %>% filter(class == class_b) %>% pull(min_score)
      max_b <- range_df %>% filter(class == class_b) %>% pull(max_score)
      percentage_in_range <- calculate_percentage_in_range(scores_a, min_b, max_b)
      percentage_matrix[i, j] <- percentage_in_range
    }else{
      percentage_matrix[i, j] <- 100
    }
  }
}

print("Percentage matrix after retraining:")
print(percentage_matrix)
###

## Testset ##
l_pred_test <- predGPsep(GPmodel_train[[OPTIMAL_GP]], test_data_X[[OPTIMAL_l]], lite = TRUE)$mean
l_mu_test <- mean(l_pred_test)
l_sd_test <- sd(l_pred_test)
print(paste0("Retrained, new class l testset mean=", l_mu_test,", sd=", l_sd_test))

other_GP_preds_mean <- lapply(existingclass_set, function(i) {
  predGPsep(GPmodel_train[[OPTIMAL_GP]], test_data_X[[paste0("c",i)]], lite = TRUE)$mean
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
  labs(title = paste0("Retrained GP", OPTIMAL_GP, " histogram of predGPsep acores by class (testset)"), x = "y-score", y = "Count") +
  theme_minimal() + 
  ylim(0, 1000)
print(histogram)

###
range_df <- preds_df %>%
  group_by(class) %>%
  summarize(min_score = min(score), max_score = max(score))

calculate_percentage_in_range <- function(scores, min_b, max_b) {
  in_range <- scores >= min_b & scores <= max_b
  percentage_in_range <- sum(in_range) / length(scores) * 100
  return(percentage_in_range)
}

# Calculate the percentage of each group's data within the range of every other group's data
classes <- unique(preds_df$class)
percentage_matrix <- matrix(0, nrow = length(classes), ncol = length(classes))
rownames(percentage_matrix) <- classes
colnames(percentage_matrix) <- classes

for (i in 1:length(classes)) {
  for (j in 1:length(classes)) {
    if (i != j) {
      class_a <- classes[i]
      class_b <- classes[j]
      scores_a <- preds_df %>% filter(class == class_a) %>% pull(score)
      min_b <- range_df %>% filter(class == class_b) %>% pull(min_score)
      max_b <- range_df %>% filter(class == class_b) %>% pull(max_score)
      percentage_in_range <- calculate_percentage_in_range(scores_a, min_b, max_b)
      percentage_matrix[i, j] <- percentage_in_range
    }else{
      percentage_matrix[i, j] <- 100
    }
  }
}

print("Percentage matrix after retraining (testset):")
print(percentage_matrix)
###

######### Train a logistic classifier for classes c and l #########
y_on_GPon <- data.frame(predGPsep(GPmodel_train[[OPTIMAL_GP]], X_on, lite = TRUE)$mean)
y_on_GPon$label <- label_on
colnames(y_on_GPon)[1] <- "value"
logistic_model <- glm(label ~ ., data = y_on_GPon, family = binomial)
print("Classifier Train Finished")
print("---------------------")

######### Test After Retrained GP_{cl} #########
test_X <- do.call(rbind, lapply(c("c0", "c1", "c2", "c3", "c4", paste0("c", l)), function(index) test_data_X[[index]]))
test_X_label <- do.call(rbind, lapply(c("c0", "c1", "c2", "c3", "c4", paste0("c", l)), function(index) test_data_label[[index]]))

test_preds <- sapply(existingclass_set, function(i) {
  predGPsep(GPmodel_train[[paste0("c", i)]], test_X, lite = TRUE)$mean
})

predicted_class_indices <- max.col(test_preds, ties.method = "first")
predicted_classes <- sapply(predicted_class_indices, function(index) {
  existingclass_set[index]
})

class_c_indices <- which(predicted_classes  == c)
if(length(class_c_indices) > 0){
  X_c <- test_X[class_c_indices, ]
  prob_class_l <- predict(logistic_model, newdata = data.frame(value = predGPsep(GPmodel_train[[OPTIMAL_GP]], X_c, lite = TRUE)$mean))
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