library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(laGP)
library(dplyr)
library(ggplot2)
library(stats)

n_tr <- 15000 # number of training data
n_ts <- 5000 # number of test data
val_fac <- 0.8 # train/validation separate factor
f <- 16 # number of features (X dimension)
nn_tr <- 400 # number of new class data, use maximum it can be if exceeds the max
d <- 10 # GP lengthscale
other_class_sample_num <- 200 # total number of other class data (whose y socre is around zero) to include in training

tau <- 0.90 # threshold used in objective function


directory_path <- paste0("Rdata_ntr", toString(n_tr), "_f", toString(f))
# pdf(file = "Rplots_Option2_filteredTraining.pdf")
pdf(file = "Rplots_GP0-4_sftmxTraining_fitZeros.pdf")

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
df <- read.csv(paste0("out/f_", toString(f), "/filtered_train_sftmx.csv"))
set.seed(430)
select.index <- sample(1:nrow(df), n_tr, replace = FALSE)
train.df <- df[select.index, ]
n_train <- floor(n_tr * val_fac)
val.df <- train.df[(n_train + 1):n_tr, ]
train.df <- train.df[1:n_train, ]

existingclass_set <- list(0, 1, 2, 3, 4)
print("Train GP0 - GP4")
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

}
print("Train Finished")
print("---------------------")

######### Prepare Test Data #########
test.df = read.csv(paste0("out/f_", toString(f), "/test_sftmx.csv"))
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

# plot [testset GPc distribution for first 5 classes]
for(c in 1:5) {
  OPTIMAL_GP <- paste0("c", c-1)

  existingclass_set <- list(0, 1, 2, 3, 4)
  GP_preds_mean <- lapply(existingclass_set, function(i) {
        predGPsep(GPmodel_train[[OPTIMAL_GP]], test_data_X[[paste0("c",i)]], lite = TRUE)$mean
  })
  GP_preds_mu <- sapply(GP_preds_mean, mean)
  GP_preds_sd <- sapply(GP_preds_mean, sd)

  x <- seq(-1, 2, length.out = 500)
  density_data <- c(
      sapply(seq_along(GP_preds_mu), function(i) dnorm(x, mean = GP_preds_mu[i], sd = GP_preds_sd[i])))
  data <- data.frame(
    x = rep(x, length(density_data)/ length(x)),
    density = density_data,
    group = factor(rep(c(paste0("class ", existingclass_set)), each = length(x))))
  plot <- ggplot(data, aes(x = x, y = density, color = group)) +
    geom_line(linewidth = 1) +
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
  histogram <- ggplot(preds_df, aes(x = score, fill = class)) +
    geom_histogram(binwidth = 0.01, alpha = 0.7, position = "identity") +
    labs(title = paste0("GP", OPTIMAL_GP, " histogram of predGPsep scores by class (testset)"), x = "y-score", y = "Count") +
    theme_minimal() + 
    ylim(0, 600)
  print(histogram)
  print(paste0("Tesetset GP",c-1," distribution for first 5 classes:"))
  print(paste0("mean: ", GP_preds_mu))
  print(paste0("sd: ", GP_preds_sd))
  print("---------------------")
}

## save to Rdata ##
save(GPmodel_train, GPresult_train, train.df, all_data_X, all_data_Y, val_data_X, val_data_Y, test_data_X, test_data_Y, file=paste0(directory_path, "/GPmodel_train_GP0-4_dmax50.Rdata"))
print("Data Saved to Directory")

######### Save Results #########
while (!is.null(dev.list()))  dev.off()