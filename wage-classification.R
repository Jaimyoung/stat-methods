library(dplyr)
library(ggplot2)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(rpart)
library(boot)
library(data.table)
library(pROC)
library(ROCR)
library(gridExtra)

adult <- tbl_df(read.csv("adult.data", header = FALSE, strip.white = TRUE))
names(adult) <-
  gsub('-', '_', c('age', 'workclass', 'fnlwgt', 'education',
                   'education-num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss',
                   'hours-per-week', 'native-country',
                   'wage'))
glimpse(adult)

# 테스트 셋으로 모형 제대로 비교하기
set.seed(1601)
n <- nrow(adult)
idx <- 1:n
training_idx <- sample(idx, n * .60)
idx <- setdiff(idx, training_idx)
validate_idx = sample(idx, n * .20)
test_idx <- setdiff(idx, validate_idx)
length(training_idx)
length(validate_idx)
length(test_idx)
training <- adult[training_idx,]
validation <- adult[validate_idx,]
test <- adult[test_idx,]


#-------------------------
# 1. GLM

ad_glm_full = glm(wage ~ ., data=training, family=binomial)
summary(ad_glm_full)
alias(ad_glm_full)
# training
# table(training$occupation)
# ad_glm_0 = glm(wage ~ 1, data=training, family=binomial)
predict(ad_glm_full, newdata = adult[1:5,], type="response")

y_obs = ifelse(validation$wage == ">50K", 1, 0)
yhat_lm = predict(ad_glm_full, newdata=validation, type='response')
binomial_deviance(y_obs, yhat_lm)
pred_lm = prediction(yhat_lm, y_obs)
perf_lm = performance(pred_lm, measure = "tpr", x.measure = "fpr")
plot(perf_lm, col='black', main="ROC Curve for GLM")
abline(0,1)
performance(pred_lm, "auc")@y.values[[1]]

#
# boxplot(split(yhat_lm, y_obs))
p1 <- ggplot(data.frame(y_obs, yhat_lm),
             aes(y_obs, yhat_lm, group=y_obs,
                 fill=factor(y_obs))) +
  geom_boxplot()
p2 <- ggplot(data.frame(y_obs, yhat_lm),
             aes(yhat_lm, fill=factor(y_obs))) +
  geom_density(alpha=.5)
grid.arrange(p1, p2, ncol=2)


#--------------
# 2. glmnet
xx = model.matrix(wage ~ .-1, adult)
x = xx[training_idx, ]
y = ifelse(training$wage == ">50K", 1, 0)
dim(x)
glimpse(x)

ad_glmnet_fit = glmnet(x, y)
plot(ad_glmnet_fit)
ad_glmnet_fit
tmp = coef(ad_glmnet_fit, s = c(.1713, .1295))
dim(tmp)
coef.glmnet()


#----
# cross validation

ad_cvfit = cv.glmnet(x, y, family = "binomial")
plot(ad_cvfit)
ad_cvfit
names(ad_cvfit)

ad_cvfit$glmnet.fit

# profile 혹은 coefficient path 를 보여준다
plot(ad_cvfit$glmnet.fit)

ad_cvfit$lambda.min

log(ad_cvfit$lambda.min)
log(ad_cvfit$lambda.1se)



coef(ad_cvfit, s=ad_cvfit$lambda.min)
coef(ad_cvfit, s="lambda.min")

coef(ad_cvfit, s=ad_cvfit$lambda.1se)
coef(ad_cvfit, s="lambda.1se")

tmp = as.matrix(coef(ad_cvfit, s="lambda.min")); tmp[tmp>0,]
tmp = as.matrix(coef(ad_cvfit, s="lambda.1se")); tmp[tmp>0,]

length(which(coef(ad_cvfit, s="lambda.min")>0))
length(which(coef(ad_cvfit, s="lambda.1se")>0))
length(which(coef(ad_cvfit, s=ad_cvfit$lambda.min)>0))
length(which(coef(ad_cvfit, s=ad_cvfit$lambda.1se)>0))

predict.cv.glmnet(ad_cvfit, s="lambda.1se", newx = x[1:5,])

predict(ad_cvfit, s="lambda.1se", newx = x[1:5,])
predict(ad_cvfit, s="lambda.1se", newx = x[1:5,], type='response')

set.seed(1607)
foldid = sample(1:10, size=length(y), replace=TRUE)
cv1 = cv.glmnet(x, y, foldid=foldid, alpha=1, family='binomial')
cv.5 = cv.glmnet(x, y, foldid=foldid, alpha=.5, family='binomial')
cv0 = cv.glmnet(x, y, foldid=foldid, alpha=0, family='binomial')

par(mfrow=c(2,2))
plot(cv1, main="Alpha=1.0 (LASSO)")
plot(cv.5, main="Alpha=0.5")
plot(cv0, main="Alpha=0.0 (Ridge)")
plot(log(cv1$lambda), cv1$cvm, pch=19, col="red",
     xlab="log(Lambda)", ylab=cv1$name, main="alpha=1.0")
points(log(cv.5$lambda), cv.5$cvm, pch=19, col="grey")
points(log(cv0$lambda), cv0$cvm, pch=19, col="blue")
legend("topleft", legend=c("alpha= 1", "alpha= .5", "alpha 0"),
       pch=19, col=c("red","grey","blue"))



y_obs = ifelse(validation$wage == ">50K", 1, 0)
yhat_glmnet = predict(ad_cvfit, s = "lambda.1se", newx=xx[validate_idx,], type='response')
# yhat_glmnet = predict(ad_cvfit, s = "lambda.min", newx=xx[validate_idx,], type='response')
yhat_glmnet = yhat_glmnet[,1] # change to a vectro from [n*1] matrix
binomial_deviance(y_obs, yhat_glmnet)
pred_glmnet = prediction(yhat_glmnet, y_obs)
perf_glmnet = performance(pred_glmnet, measure = "tpr", x.measure = "fpr")
plot(perf_lm, col='black', main="ROC Curve")
plot(perf_glmnet, col='blue', add=TRUE)
abline(0,1)
legend('bottomright', inset=.1,
       legend = c("GLM", "glmnet"),
       col=c('black', 'blue'), lty=1, lwd=2)
performance(pred_glmnet, "auc")@y.values[[1]]



#-------------------
# 3. 트리
cvr_tr <- rpart(wage ~ ., data = training)
cvr_tr

#cvr_tr <- rpart(wage ~ ., data = adult, control = rpart.control(cp=0.001))
#cvr_tr
#?rpart.control

printcp(cvr_tr)
summary(cvr_tr)

opar = par(mfrow = c(1,1), xpd = NA)
# otherwise on some devices the text is clipped
plot(cvr_tr)
text(cvr_tr, use.n = TRUE)
par(opar)



yhat_tr = predict(cvr_tr, validation)
yhat_tr = yhat_tr[,">50K"]
binomial_deviance(y_obs, yhat_tr)
pred_tr = prediction(yhat_tr, y_obs)
perf_tr = performance(pred_tr, measure = "tpr", x.measure = "fpr")
plot(perf_lm, col='black', main="ROC Curve")
plot(perf_tr, col='blue', add=TRUE)
abline(0,1)
legend('bottomright', inset=.1,
       legend = c("GLM", "Tree"),
       col=c('black', 'blue'), lty=1, lwd=2)
performance(pred_tr, "auc")@y.values[[1]]




#---------------
# 4. 랜덤 포레스트
set.seed(1607)
ad_rf = randomForest(wage ~ ., training)
ad_rf
plot(ad_rf)

tmp <- importance(ad_rf)
head(round(tmp[order(-tmp[,1]), 1, drop=FALSE], 2), n=10)
varImpPlot(ad_rf)

predict(ad_rf, newdata = adult[1:5,])
predict(ad_rf, newdata = adult[1:5,], type="prob")



yhat_rf = predict(ad_rf, newdata = validation, type='prob')[,'>50K']
binomial_deviance(y_obs, yhat_rf)
pred_rf = prediction(yhat_rf, y_obs)
perf_rf = performance(pred_rf, measure = "tpr", x.measure = "fpr")
plot(perf_lm, col='black', main="ROC Curve")
plot(perf_glmnet, add=TRUE, col='blue')
plot(perf_rf, add=TRUE, col='red')
abline(0,1)
legend('bottomright', inset=.1,
       legend = c("GLM", "glmnet", "RF"),
       col=c('black', 'blue', 'red'), lty=1, lwd=2)
performance(pred_tr, "auc")@y.values[[1]]


p1 <- data.frame(yhat_glmnet, yhat_rf) %>%
  ggplot(aes(yhat_glmnet, yhat_rf)) +
  geom_point(alpha=.5) +
  geom_abline() +
  geom_smooth()
p2 <- reshape2::melt(data.frame(yhat_glmnet, yhat_rf)) %>%
  ggplot(aes(value, fill=variable)) +
  geom_density(alpha=.5)
grid.arrange(p1, p2, ncol=2)



#-----------------
# 5. GBM

set.seed(1607)
adult_gbm = training %>% mutate(wage=ifelse(wage == ">50K", 1, 0))
ad_gbm = gbm(wage ~ ., data=adult_gbm,
             distribution = "bernoulli",
             n.trees=50000, cv.folds=3, verbose = TRUE)
(best_iter = gbm.perf(ad_gbm, method="cv"))

ad_gbm2 = gbm.more(ad_gbm, n.new.trees = 10000)
(best_iter = gbm.perf(ad_gbm2, method="cv"))


predict(ad_gbm, n.trees=best_iter, newdata=adult_gbm[1:5,], type='response')

ad_gbm$cv.error[best_iter]



plot(ad_gbm)


yhat_gbm = predict(ad_gbm, n.trees=best_iter, newdata=validation, type='response')
binomial_deviance(y_obs, yhat_gbm)
pred_gbm = prediction(yhat_gbm, y_obs)
perf_gbm = performance(pred_gbm, measure = "tpr", x.measure = "fpr")
plot(perf_lm, col='black', main="ROC Curve")
plot(perf_glmnet, add=TRUE, col='blue')
plot(perf_rf, add=TRUE, col='red')
plot(perf_gbm, add=TRUE, col='cyan')
abline(0,1)
legend('bottomright', inset=.1,
       legend = c("GLM", "glmnet", "RF", "GBM"),
       col=c('black', 'blue', 'red', 'cyan'), lty=1, lwd=2)
performance(pred_gbm, "auc")@y.values[[1]]


a <- data.frame(lm = performance(pred_lm, "auc")@y.values[[1]],
           glmnet = performance(pred_glmnet, "auc")@y.values[[1]],
           rf = performance(pred_rf, "auc")@y.values[[1]],
           gbm = performance(pred_gbm, "auc")@y.values[[1]])  %>%
  reshape2::melt(value.name = 'auc', variable.name = 'method')

b <- data.frame(lm = binomial_deviance(y_obs, yhat_lm),
           glmnet = binomial_deviance(y_obs, yhat_glmnet),
           rf = binomial_deviance(y_obs, yhat_rf),
           gbm = binomial_deviance(y_obs, yhat_gbm)) %>%
  reshape2::melt(value.name = 'binomial_deviance', variable.name = 'method')
a %>% inner_join(b)


y_obs_test = ifelse(test$wage == ">50K", 1, 0)
yhat_gbm_test = predict(ad_gbm, n.trees=best_iter, newdata=test, type='response')
binomial_deviance(y_obs_test, yhat_gbm_test)
pred_gbm_test = prediction(yhat_gbm_test, y_obs_test)
performance(pred_gbm_test, "auc")@y.values[[1]]





# exmaple(pairs) 에서 따옴
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}

pairs(data.frame(y_obs=y_obs,
                 yhat_lm=yhat_lm,
                 yhat_glmnet=c(yhat_glmnet),
                 yhat_rf=yhat_rf,
                 yhat_gbm=yhat_gbm),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)



#--------------------
binomial_deviance = function(y_obs, yhat){
  epsilon = 0.0001
  yhat = ifelse(yhat < epsilon, epsilon, yhat)
  yhat = ifelse(yhat > 1-epsilon, 1-epsilon, yhat)
  a = ifelse(y_obs==0, 0, y_obs * log(y_obs/yhat))
  b = ifelse(y_obs==1, 0, (1-y_obs) * log((1-y_obs)/(1-yhat)))
  return(2*sum(a + b))
}

