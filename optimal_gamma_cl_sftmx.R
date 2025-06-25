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
nn_tr <- 100 # number of new class data, use maximum it can be if exceeds the max


tau <- 0.90 # threshold used in objective function
pen_scale <- 100  # penalty scale on GP original class (e.g class 4 for GP4)

pdf(file = "optimal_gamma_cl_dmax50.pdf")

# load RData
load(paste0("Rdata_ntr", n_tr, "_f", f, "/GPmodel_train_GP0-4_dmax50.RData"))

######### Add New Class (label 5-9) #########
print("Add New Class")
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
    # if(label == l){ 
    #   NNc_labell_Y <- select_newClass[, (f + c + 1), drop = FALSE]
    # }

    # select_newClass_val <- as.matrix(val.df[val.df[, "label"] == label, ])
    # val_data_X[[key]] <- select_newClass_val[, 1:f]
    # val_data_Y[[key]] <- select_newClass_val[, (f + label + 1), drop = FALSE]

    # calculate sd from nn
    NNscores_sd[[key]] <- sd(all_data_Y[[key]])
    #print(NNscores_sd[[key]])
}

#select new_class l and optimal yy for each target_model c
for(c in 1:5) { # each c
    GPcc_out <-  predGPsep(GPmodel_train[[paste0("c", c-1)]], all_data_X[[paste0("c", c-1)]], lite=TRUE)
    mean_cc <- GPcc_out$mean
    for(l in 6:10){ # each l
        objective_fn <- function(yy){
            value <- 0
            for(i in 1:5){ # other exisiting classes except l
                pen <- 100
                if(i == c){
                  pen <- pen_scale
                }
                X_i <- all_data_X[[paste0("c", i-1)]]
                GP_out <- predGPsep(GPmodel_train[[paste0("c", c-1)]], X_i, lite=TRUE)
                mean <- GP_out$mean
                mu <- mean(mean)
                sigma <- sd(mean)
                # find region of prob >= tau
                alpha <- (1 - tau)/2 
                lower_bound <- qnorm(alpha, mean = mu, sd = sigma)
                upper_bound <- qnorm(1-alpha, mean = mu, sd = sigma)
                prob <- pnorm(upper_bound, mean = yy, sd = NNscores_sd[[paste0("c",l-1)]]) - pnorm(lower_bound, mean = yy, sd = NNscores_sd[[paste0("c",l-1)]])
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
