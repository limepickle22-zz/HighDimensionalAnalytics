> library(caret)
> library(glmnet)
> library(randomForest)
> cancertrain = read.csv("C:\\Users\\SSamtani\\Downloads\\cancer.train.csv", header = FALSE)
> cancertrain = data.frame(scale(as.matrix(cancertrain)))
> labeltrain = read.csv("C:\\Users\\SSamtani\\Downloads\\label.train.csv", header = FALSE)
> labeltrain = ifelse(labeltrain == 'M', 1, 0)
> cancertrain = cbind(labeltrain, cancertrain)
> colnames(cancertrain)[1] = "y"
> cancertest = read.csv("C:\\Users\\SSamtani\\Downloads\\cancer-1.test.csv", header = FALSE)
> cancertest = data.frame(scale(as.matrix(cancertest)))
> labeltest = read.csv("C:\\Users\\SSamtani\\Downloads\\label.test.csv", header = FALSE)
> labeltest = as.factor(ifelse(labeltest == 'M', 1, 0))
#################### PART B ######################################
> logit = glm(y ~., data = cancertrain, family = "binomial", maxit = 100)
> predictions = as.factor(round(predict.glm(logit, newdata = cancertest, type = "response"), 0))
> cm = confusionMatrix(data = predictions, reference = labeltest, positive = "1")
> draw_confusion_matrix <- function(cm) {
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)

  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'B', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'M', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'B', cex=1.2, srt=90)
  text(140, 335, 'M', cex=1.2, srt=90)

  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')

  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)

  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  
> draw_confusion_matrix(cm)
################## PART C #########################################
> cvridge = cv.glmnet(as.matrix(cancertrain[, -1]), as.factor(cancertrain[, 1]), family = "binomial", alpha = 0)
> optlamridge = cvridge$lambda.min
> optlamridge
> ridge = glmnet(as.matrix(cancertrain[, -1]), as.matrix(cancertrain[, 1]), alpha = 0, family = "binomial", lambda = optlamridge)
> ridge$beta
> predictRidge = as.factor(round(predict(ridge, s = optlamridge, newx = as.matrix(cancertest), type = "response"), 0))
> cmridge = confusionMatrix(data = predictRidge, reference = labeltest, positive = "1")
> draw_confusion_matrix(cmridge)
################## PART D #########################################
> cvlasso = cv.glmnet(as.matrix(cancertrain[, -1]), as.matrix(cancertrain[, 1]), alpha = 1, family = "binomial")
> optlamlasso = cvlasso$lambda.min
> optlamlasso
> lasso = glmnet(as.matrix(cancertrain[, -1]), as.matrix(cancertrain[, 1]), alpha = 1, lambda = optlamlasso)
> lasso$beta
> predictLasso = as.factor(round(predict(lasso, s = optlamlasso, newx = as.matrix(cancertest), type = "response"), 0))
> cmlasso = confusionMatrix(data = predictLasso, reference = labeltest, positive = "1")
> draw_confusion_matrix(cmlasso)
################## PART E #########################################
> gamma = 1
> penalty = 1/(abs(matrix(coef(cvridge, s = cvridge$lambda.min))[2:(ncol(cancertrain))]))^gamma
> cvalasso = cv.glmnet(as.matrix(cancertrain)[, -1], as.matrix(cancertrain)[, 1], alpha = 1, family = "binomial", penalty.factor = penalty)
> optlamalasso = cvalasso$lambda.min 
> optlamalasso 
> alasso = glmnet(as.matrix(cancertrain)[, -1], as.matrix(cancertrain)[, 1], alpha = 1, family = "binomial", lambda = optlamalasso, penalty.factor = 1)
> alasso$beta 
> predictAlasso = as.factor(round(predict(alasso, s = optlamalasso, newx = as.matrix(cancertest), type = "response"), 0))
> cmalasso = confusionMatrix(data = predictAlasso, reference = labeltest, positive = "1")
> draw_confusion_matrix(cmalasso)
################## PART F #########################################
> randomForest(as.factor(y) ~., data = cancertrain, xtest = cancertest, ytest = labeltest, ntree = 1000)$test$confusion
