> library(caret)
> library(glmnet)
> library(glmnetUtils)
> set.seed(0)
> train = read.csv("C:\\Users\\SSamtani\\Downloads\\train.air.csv", header = TRUE)
> test = read.csv("C:\\Users\\SSamtani\\Downloads\\test.air.csv", header = TRUE)
> pretrain = preProcess(train, method = c("center", "scale"))
> pretest = preProcess(test, method = c("center", "scale"))
> trainstan = predict(pretrain, train)
> teststan = predict(pretest, test)
> trainstan = as.matrix(trainstan)
> teststan = as.matrix(teststan)
> cvridge = cv.glmnet(trainstan[, -1], trainstan[, 1], alpha = 0)
> optlamridge = cvridge$lambda.min
> optlamridge
> ridge = glmnet(trainstan[, -1], trainstan[, 1], alpha = 0, lambda = optlamridge)
> ridge$beta
> predictCOridge = predict(ridge, s = optlamridge, newx = teststan[, -1])
> ridgeMSPE = mean((teststan[, 1] - predictCOridge) ^ 2)
> ridgeMSPE 
> cvlasso = cv.glmnet(trainstan[, -1], trainstan[, 1], alpha = 1)
> optlamlasso = cvlasso$lambda.min
> optlamlasso
> lasso = glmnet(trainstan[, -1], trainstan[, 1], alpha = 1, lambda = optlamlasso)
> lasso$beta
> predictCOlasso = predict(lasso, s = optlamlasso, newx = teststan[, -1])
> lassoMSPE = mean((teststan[, 1] - predictCOlasso) ^ 2)
> lassoMSPE 
> gamma = 1
> penalty = 1/(abs(matrix(coef(cvridge, s = cvridge$lambda.min))[2:(ncol(trainstan))]))^gamma
> cvalasso = cv.glmnet(trainstan[, -1], trainstan[, 1], alpha = 1, family = "gaussian", penalty.factor = penalty)
> optlamalasso = cvalasso$lambda.min 
> optlamalasso 
> alasso = glmnet(trainstan[, -1], trainstan[, 1], alpha = 1, family = "gaussian", lambda = optlamalasso, penalty.factor = 1)
> alasso$beta 
> predictCOalasso = predict(alasso, s = optlamalasso, newx = teststan[, -1])
> alassoMSPE = mean((teststan[, 1] - predictCOalasso)^2)
> alassoMSPE 
> a = seq(0.1, .9, .05)
> search = foreach(i = a, .combine = rbind) %dopar% {
  + cv = cv.glmnet(trainstan[, -1], trainstan[, 1], family = "gaussian", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = i)
  + data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
+ }
> tuneal = search[search$cvm == min(search$cvm),]
> tuneal$alpha 
> tuneal$lambda 
> elastic = glmnet(trainstan[, -1], trainstan[, 1], alpha = tuneal$alpha, lambda = tuneal$lambda.1se)
> elastic$beta
> predictCOelastic = predict(elastic, s = tuneal$lambda, newx = teststan[, -1])
> elasticMSPE = mean((teststan[, 1] - predictCOelastic) ^ 2)
> elasticMSPE
