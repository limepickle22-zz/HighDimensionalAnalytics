> library(R.matlab)
> library(splines)
> library(caret)
> library(gglasso)
> NSCTrain = readMat("C:\\Users\\SSamtani\\Downloads\\NSC.mat", header = TRUE)
> NSCTest = readMat("C:\\Users\\SSamtani\\Downloads\\NSC.test.mat", header = TRUE)
> yTrain = data.frame(NSCTrain['y'])
> xTrain = data.frame(NSCTrain['x'])
> plot(seq(1, 203), xTrain[1:1,1:203], xlab = 'Time', ylab = 'Measurement', main = 'Air Aspirated per Cylinder')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 1:203])
+ }
> plot(seq(1, 203), xTrain[1:1,204:406], ylim = c(1530, 1600), xlab = 'Time', ylab = 'Measurement', main = 'Engine Rotational Speed')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 204:406])
+ }
> plot(seq(1, 203), xTrain[1:1,407:609], xlab = 'Time', ylab = 'Measurement', ylim = c(0, 50), main = 'Total Quantity of Fuel Injected')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 407:609])
+ }
> plot(seq(1, 203), xTrain[1:1,610:812], xlab = 'Time', ylab = 'Measurement', main = 'Low Pressure EGR Valve')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 610:812])
+ }
> plot(seq(1, 203), xTrain[1:1,813:1015], xlab = 'Time', ylab = 'Measurement', ylim = c(140, 230), main = 'Inner Torque')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 813:1015])
+ }
> plot(seq(1, 203), xTrain[1:1, 1016:1218], xlab = 'Time', ylab = 'Measurement', ylim = c(20, 70), main = 'Accelerator Pedal Position')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 1016:1218])
+ }
> plot(seq(1, 203), xTrain[1:1, 1219:1421], xlab = 'Time', ylab = 'Measurement', ylim = c(-25, 25), main = 'Aperture Ration of Inlet Valve')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 1219:1421])
+ }
> plot(seq(1, 203), xTrain[1:1, 1422:1624], xlab = 'Time', ylab = 'Measurement', ylim = c(1500, 1600), main = 'Downstream Intercooler Pressure')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 1422:1624])
+ }
> plot(seq(1, 203), xTrain[1:1, 1625:1827], xlab = 'Time', ylab = 'Measurement', ylim = c(0, 50), main = 'Fuel in the Second Pre-Injection')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 1625:1827])
+ }
> plot(seq(1, 203), xTrain[1:1, 1828:2030], xlab = 'Time', ylab = 'Measurement', ylim = c(50, 100), main = 'Vehicle Velocity')
> for (i in 1:150) {
+ lines(seq(1, 203), xTrain[i:i, 1828:2030])
+ }
> preyTrain = preProcess(yTrain, method = c("center", "scale"))
> prexTrain = preProcess(xTrain, method = c("center", "scale"))
> yTrain = predict(preyTrain, yTrain)
> xTrain = predict(prexTrain, xTrain)
> YTrain = matrix(nrow = 150, ncol = 11)
> for (i in 1:150) {
+ data = data.frame("time" = seq(1, 203), "value" = as.numeric(yTrain[i:i, 1:203]))
+ coefficients = coef(lm(value ~ bs(time, df = 11), data = data))
+ coefficients = as.numeric(coefficients[2:length(coefficients)])
+ YTrain[i, ] = coefficients
+ }
> XTrain = matrix(nrow = 150, ncol = 110)
> for (j in 1:10) {
+ for (i in 1:150) {
+ data = data.frame("time" = seq(1, 203), "value" = as.numeric(xTrain[i:i, (1 + 203*(j-1)):(203*j)]))
+ coefficients = coef(lm(value ~ bs(time, df = 11), data = data))
+ coefficients = as.numeric(coefficients[2:length(coefficients)])
+ XTrain[i, (1 + 11*(j-1)):(11*j)] = coefficients
+ }
+ }
> YTrainVec = as.vector(YTrain)
> XTrainVec = kronecker(diag(11), XTrain)
> model = gglasso(XTrainVec, YTrainVec, group = rep(1:10, each = 121), lambda = .0037)
> model$beta 
> yTest = data.frame(NSCTest['y.test'])
> xTest = data.frame(NSCTest['x.test'])
> preyTest = preProcess(yTest, method = c("center", "scale"))
> prexTest = preProcess(xTest, method = c("center", "scale"))
> yTest = predict(preyTest, yTest)
> xTest = predict(prexTest, xTest)
> YTest = matrix(nrow = 50, ncol = 11)
> for (i in 1:50) {
+ data = data.frame("time" = seq(1, 203), "value" = as.numeric(yTest[i:i, 1:203]))
+ coefficients = coef(lm(value ~ bs(time, df = 11), data = data))
+ coefficients = as.numeric(coefficients[2:length(coefficients)])
+ YTest[i, ] = coefficients
+ }
> XTest = matrix(nrow = 50, ncol = 110)
> for (j in 1:10) {
+ for (i in 1:50) {
+ data = data.frame("time" = seq(1, 203), "value" = as.numeric(xTest[i:i, (1 + 203*(j-1)):(203*j)]))
+ coefficients = coef(lm(value ~ bs(time, df = 11), data = data))
+ coefficients = as.numeric(coefficients[2:length(coefficients)])
+ XTest[i, (1 + 11*(j-1)):(11*j)] = coefficients
+ }
+ }
> YTestVec = as.vector(YTest)
> XTestVec = kronecker(diag(11), XTest)
> predictions = predict.gglasso(model, newx = XTestVec)
> MSPE = mean((YTestVec - predictions) ^ 2)


