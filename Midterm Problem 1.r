> library(splines)
> library(fda)
> library(randomForest)
> library(Metrics)
> X = read.table("C:\\Users\\SSamtani\\Downloads\\X.txt", header = FALSE)
> Y = read.table("C:\\Users\\SSamtani\\Downloads\\Y.txt", header = FALSE)
> TrainX = X$V1[1:80]
> TrainY = Y[1:80, ]
> TestX = X$V1[81:100]
> TestY = Y[81:100, ]
> smooth = matrix(nrow = 80, ncol = length(smooth.spline(as.numeric(TrainY[1,]))$fit$coef))
> smoothTest = matrix(nrow = 20, ncol = length(smooth.spline(as.numeric(TrainY[1,]))$fit$coef))
> for (i in 1:80) {
+ smooth[i, ] = smooth.spline(as.numeric(TrainY[i,]))$fit$coef
+ }
> for (i in 1:20) {
+ smoothTest[i, ] = smooth.spline(as.numeric(TestY[i,]))$fit$coef 
+ }
> TrainXSmooth = data.frame("TTF" = TrainX, smooth)
> TestXSmooth = data.frame(smoothTest)
> numpc = 2
> fdo = Data2fd(argvals = seq(1, 128), y = t(as.matrix(TrainY)))
> fdoTest = Data2fd(argvals = seq(1, 128), y = t(as.matrix(TestY)))
> fpca = pca.fd(fdo, nharm = numpc)$scores 
> fpcaTest = pca.fd(fdoTest, nharm = numpc)$scores 
> TrainXFPCA = data.frame("TTF" = TrainX, fpca)
> TestXFPCA = data.frame(fpcaTest)
> head(TrainXSmooth)
> head(TrainXFPCA)
> pairs(TrainXSmooth[1:10])
> pairs(TrainXFPCA)
> smoothRF = randomForest(TrainX ~., data = TrainXSmooth, xtest = TestXSmooth)
> fpcaLinModel = lm(TrainX ~., data = TrainXFPCA)
> summary(fpcaLinModel)
> smoothRF$test$predicted
> rmse(TestX, smoothRF$test$predicted)
> predict.lm(fpcaLinModel, newdata = TestXFPCA)
> rmse(TestX, predict.lm(fpcaLinModel, newdata = TestXFPCA))

