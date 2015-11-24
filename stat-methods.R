library(dplyr)
library(ggplot2)
library(gridExtra)

glimpse(mpg)
glimpse(diamonds)
glimpse(diamonds)
?lm

#----------
# 1. continuous variable


?qqplot
#----------
# 1. 중요한 확률분포들. 정규분포와 이항분포.
rbinom(1, 5, 1/4)
rbinom(1, 5, 1/4)
rbinom(1, 5, 1/4)

set.seed(1510)
x = rbinom(1000, 5, 1/4)
table(x)
plot(table(x)) # 도수분포
plot(prop.table(table(x))) # 상대도수분포
lines(0:5, dbinom(0:5, 5, 1/4), col='red')
summary(x)
c(mean(x), sd(x)) # 표본평균과 표준편차
c(5 * 1/4, sqrt(5 * 1/4 * 3/4)) # 이론값: E(X), SD(X)



set.seed(1510)
x = rnorm(1000, 5, 2)
plot(density(x))
hist(x, probability = TRUE)
lines(density(x), col='blue', lty=2) # 커널 density estimate
curve(dnorm(x, 5, 2), col='red', add=TRUE) # 이론분포

summary(x)
c(mean(x), sd(x)) # 표본평균과 표준편차
c(5, 2) # 이론값: E(X), SD(X)

?Distributions
rpois()
rexp()


#---------x
# 1. lm

## Annette Dobson (1990) "An Introduction to Generalized Linear Models".
## Page 9: Plant Weight Data.
ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
weight <- c(ctl, trt)
lm.D9 <- lm(weight ~ group)
lm.D90 <- lm(weight ~ group - 1) # omitting intercept

anova(lm.D9)
summary(lm.D90)

opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(lm.D9, las = 1)      # Residuals, Fitted, ...
par(opar)

### less simple examples in "See Also" above


#---------------
# 0. 모든 자료에 대한 분석
glimpse(mpg)
pairs(mpg)

str(mpg)
head(mpg)
summary(mpg)

#-------------
# 1. 일변량 연속형 변수
summary(mpg$hwy)
mean(mpg$hwy)
median(mpg$hwy)
range(mpg$hwy)
quantile(mpg$hwy)

old_par = par(mfrow=c(2,2))
hist(mpg$hwy)
boxplot(mpg$hwy)
qqnorm(mpg$hwy)
qqline(mpg$hwy)
par(old_par)





hwy = mpg$hwy
n = length(hwy)
mu0 = 22.9
t.test(hwy, mu=mu0, alternative = "greater")

# 신뢰구간을 수식으로 계산하기
c(mean(hwy) - qt(.975, df = n-1) * sd(hwy)/sqrt(n),
  mean(hwy) + qt(.975, df = n-1) * sd(hwy)/sqrt(n))

# p-value를 수식으로 계산하기
(t_stat = (mean(hwy) - mu0)/(sd(hwy)/sqrt(n)))
1 - pt(t_stat, df = n-1)


# 로버스트 통계량
c(mean(hwy), sd(hwy))
c(median(hwy), mad(hwy))


#-------------
# 2. 일변량 범주형 변수
glimpse(mpg)

(t1 = table(diamonds$cut))
(t2 = xtabs(~ cut, diamonds))
prop.table(t1)
class(t1)
class(t2)

ps = c(3, 9, 22, 25, 40)
(ps = ps / sum(ps))
chisq.test(t1, p = ps)

(obs = t1)
(expected = ps * sum(obs))
chisq_stat = sum((obs - expected)^2 / expected)
1 - pchisq(chisq_stat, df = length(obs) - 1)


old_par = par(mfrow=c(1,2))
plot(t2)
mosaicplot(t2)
par(old_par)



#-------------
# 3. 연속형 변수 x, y
glimpse(mpg)

cor(mpg$cty, mpg$hwy)
with(mpg, cor(cty, hwy))
with(mpg, cor(cty, hwy, method = "kendall"))
with(mpg, cor(cty, hwy, method = "spearman"))

ggplot(mpg, aes(cty, hwy)) + geom_jitter() +
  geom_smooth(method="lm")


(hwy_lm = lm(hwy ~ cty, data=mpg))
summary(hwy_lm)


predict(hwy_lm)
resid(hwy_lm)
predict(hwy_lm, newdata = data.frame(cty=c(10, 20, 30)))
predict(hwy_lm, newdata = data.frame(cty=c(10, 20, 30)),
        se.fit=TRUE)



class(hwy_lm)
opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(hwy_lm, las = 1)      # Residuals, Fitted, ...
par(opar)


# 로버스트 회귀분석
library(MASS)
set.seed(123) # make reproducible
lqs(stack.loss ~ ., data = stackloss) # 로버스트
lm(stack.loss ~ ., data = stackloss) # 보통 선형모형

??design
?model.matrix
?contrasts
plot(stackloss)


options("contrasts")
levels(mpg$class)

?lm
#-------------
# 4. 범주형 x, 연속형 y

mean(mpg$hwy)
ggplot(mpg, aes(class, hwy)) + geom_boxplot()

(hwy_lm2 = lm(hwy ~ class, data=mpg))
summary(hwy_lm2)


predict(hwy_lm2, newdata=data.frame(class="pickup"))
coef(hwy_lm2)
hwy_lm2$coefficients
hwy_lm2$coefficients["(Intercept)"] +
  hwy_lm2$coefficients["classpickup"]


opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(hwy_lm2, las = 1)      # Residuals, Fitted, ...
par(opar)


#--------------------
# 5. 범주형+연속형 x, 연속형 y

(hwy_lm3 = lm(hwy ~ class + cty, data=mpg))
summary(hwy_lm3)

predict(hwy_lm3, newdata=data.frame(class="pickup", cty=22))

opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(hwy_lm3, las = 1)      # Residuals, Fitted, ...
par(opar)
