# For each existing(trained) target_modle c (GP0 - GP4), we find a new_class with label l to add to model c;
# use an objective function to find optimal GP to retrain and its corresponding yy;
# train a logistic classifier to separate target_model class and new class
library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(laGP)
library(dplyr)
library(ggplot2)
library(stats)

n_tr <- 15000
n_ts <- 5000
val_fac <- 0.8
f <- 16
nn_tr <- 50 # number of new class data, use maximum it can be if exceeds the max


tau <- 0.90 # threshold used in objective function
pen_scale <- 100  # penalty scale on GP original class (e.g class 4 for GP4)

pdf(file = "optimal_gamma_cl.pdf")

# Training dataset
df <- read.csv(paste0("out/f_", toString(f), "/filtered_train_sftmx.csv")) # [,-1]
set.seed(430)
select.index <- sample(1:nrow(df), n_tr, replace = FALSE)
train.df <- df[select.index, ]
# Add validation set
n_train <- floor(n_tr * val_fac)
val.df <- train.df[(n_train + 1):n_tr, ]
train.df <- train.df[1:n_train, ]

# Train GP0 to GP4
all_data_X <- vector("list", 10)
GPmodel_train <- vector("list", 5)
GPresult_train <- vector("list", 5)
mse_train <- vector("list", 5)
for (j in 1:5) {
  X <- as.matrix(train.df[train.df[, "label"] == j - 1, 1:f])
  Y <- matrix(train.df[train.df[, "label"] == j - 1, (f + j)])
  val.X <- as.matrix(val.df[val.df[, "label"] == j - 1, 1:f])
  val.Y <- matrix(val.df[val.df[, "label"] == j - 1, (f + j)])

  gpisep <- newGP(X, Y, d = 10, g = 1e-4, dK = TRUE)
  out <- predGP(gpisep, X, lite = TRUE)
  all_data_X[[j]] <- X
  GPresult_train[[j]] <- out
  GPmodel_train[[j]] <- gpisep

  val_result <- predGP(gpisep, val.X, lite = TRUE)
  mse <- norm(val_result$mean - val.Y, "2")
  print(mse)
  mse_train[[j]] <- mse
  plot(val.Y, val_result$mean, ylab = "Y_pred", xlab = "Y_true")
  title(paste0("Validation True vs. Pred (class=", j, ")"))
  rm(gpisep)
}
print("Train Finished")
print("---------------------")

# Test1 for first 5 classes (label = 0,1,2,3,4)
test.df = read.csv(paste0("out/f_", toString(f), "/test_sftmx.csv"))
test.df <- test.df[1:n_ts, ]
test.df <- test.df[test.df[, "label"] < 5, ]
test.X <- as.matrix(test.df[, 1:f])
  
print(paste("Test1 Data: ", nrow(test.df[test.df[, "label"] < 5, ]), " from classes label 0-4"))
GP_test <- vector("list", 10)
GP_test_mean <- vector("list", 10)
for (j in 1: 5) {
  gpisep <- GPmodel_train[[j]]
  out <- predGP(gpisep, test.X, lite=TRUE)
  GP_test[[j]] <- out
  GP_test_mean[[j]] <- out$mean
}
test_result <- do.call(cbind, GP_test_mean)
test_result_labels <- test.df["label"]

correct_predictions <- sapply(1:nrow(test_result), function(i) {
  which.max(test_result[i, ]) == (test_result_labels[i, 1] + 1)
})
test1_accuracy <- mean(correct_predictions)
print(paste0("Test1 first 5 classes test accuracy is ", test1_accuracy))

class_labels <- 0:4
accuracy_per_label <- sapply(class_labels, function(label) {
  label_indices <- which(test_result_labels[, 1] == label)
  print(paste0("Class Label ", label, " recall is ", mean(correct_predictions[label_indices])))
})
print("---------------------")

# New_class data (label 5-9)
NNscores_sd <- vector("list", 5)
for(j in 1:5){
    label <- j + 4
    num_rows <- nrow(train.df[train.df[, "label"] == label, ])
    nn_tr <- min(nn_tr, num_rows)
    print(paste0("New class ", label, " has ", nn_tr, " rows of data"))
    indices <- sample(num_rows, nn_tr, replace = FALSE)
    select_newClass <- as.matrix(train.df[train.df[, "label"] == label, ])
    select_newClass <- select_newClass[indices, ]
    all_data_X[[j + 5]] <- select_newClass[, 1:f]
    # calculate sd from nn
    NNscores <- select_newClass[, (f + 1 + label), drop = FALSE]
    NNscores_sd[[j]] <- sd(NNscores)
}

#select new_class l and optimal yy for each target_model c
for(c in 1:5) { # each c
    GPcc_out <-  predGP(GPmodel_train[[c]], all_data_X[[c]], lite=TRUE)
    mean_cc <- GPcc_out$mean
    for(l in 6:10){ # each l
        objective_fn <- function(yy){
            value <- 0
            for(i in 1:5){ # other exisiting classes except l
                pen <- 100
                if(i == c){
                  pen <- pen_scale
                }
                X_i <- all_data_X[[i]]
                GP_out <- predGP(GPmodel_train[[c]], X_i, lite=TRUE)
                mean <- GP_out$mean
                mu <- mean(mean)
                sigma <- sd(mean)
                # find region of prob >= tau
                alpha <- (1 - tau)/2 
                lower_bound <- qnorm(alpha, mean = mu, sd = sigma)
                upper_bound <- qnorm(1-alpha, mean = mu, sd = sigma)
                prob <- pnorm(upper_bound, mean = yy, sd = NNscores_sd[[l-5]]) - pnorm(lower_bound, mean = yy, sd = NNscores_sd[[l-5]])
                value <- value + pen * prob
            }
            return(value)
        }

        op_result <- optim(
            par = 0.9,
            fn = objective_fn,
            method = "L-BFGS-B",
            lower = 0,
            upper = mean_cc
        )

        print(paste0("GP", c-1, " l=" , l-1 , " optimal yy=", op_result$par))
        print(paste0("GP", c-1, " l=" , l-1 ," objective value=", op_result$value))  
    }
}

print("---------------------")

# for(j in 1:5){
#   print(paste0("class ", j+4, " sd = ", NNscores_sd[[j]]))
# }

# Plot, example for GP4
# print("Plot an Example - GP1")
# X_0 <- all_data_X[[1]]
# X_1 <- all_data_X[[2]]
# X_2 <- all_data_X[[3]]
# X_3 <- all_data_X[[4]]
# X_4 <- all_data_X[[5]] 
# GP1_X0 <- predGP(GPmodel_train[[2]], X_0, lite=TRUE) # GP4(X0)
# GP1_X1 <- predGP(GPmodel_train[[2]], X_1, lite=TRUE) # GP4(X1)
# GP1_X2 <- predGP(GPmodel_train[[2]], X_2, lite=TRUE) # GP4(X2)
# GP1_X3 <- predGP(GPmodel_train[[2]], X_3, lite=TRUE) # GP4(X3)
# GP1_X4 <- predGP(GPmodel_train[[2]], X_4, lite=TRUE) # GP4(x4)

# print(paste0("GP1(X1)  mean of GP$mean =", mean(GP1_X1$mean), ", sd of GP$mean =", sd(GP1_X1$mean)))
# print(paste0("GP1(X0) mean of GP$mean =", mean(GP1_X0$mean), ", sd of GP$mean =", sd(GP1_X0$mean)))

# x <- seq(-10, 15, length.out = 500)
# data <- data.frame(
#   x = rep(x, 6),
#   density = c(dnorm(x, mean = mean(GP1_X0$mean), sd = sd(GP1_X0$mean)), 
#               dnorm(x, mean = mean(GP1_X1$mean), sd = sd(GP1_X1$mean)), 
#               dnorm(x, mean = mean(GP1_X2$mean), sd = sd(GP1_X2$mean)), 
#               dnorm(x, mean = mean(GP1_X3$mean), sd = sd(GP1_X3$mean)), 
#               dnorm(x, mean = mean(GP1_X4$mean), sd = sd(GP1_X4$mean)), 
#               dnorm(x, mean = 3.99997530560347, sd = NNscores_sd[[3]])),
#   group = factor(rep(c("class 0", "class 1", "class 2", "class 3", "class 4", "class 7 with mean=3.99997530560347"), each = length(x)))
# )

# # Plot using ggplot2
# plot <- ggplot(data, aes(x = x, y = density, color = group)) +
#   geom_line(size = 1) +
#   labs(title = "GP1 output distribution on different classes", x = "y-score", y = "Density") +
#   theme_minimal() + 
#   ylim(0, 0.1)


# print(plot)
print("done")
while (!is.null(dev.list()))  dev.off()
