traindata = read.csv("train.csv")
testdata = read.csv("test.csv")
library(dplyr)
library(fastDummies)
library(caret)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(grid)
library(tidyr)

head(traindata)

head(testdata)


#complete_data <- rbind(traindata, testdata)

#train data preprocessing
summary(traindata)

#Data profiling
sapply(traindata, function(x) length(unique(x)))

unique_x <- unique(traindata)

length(x) != length(unique_x)

colSums(is.na(traindata))
colSums(traindata=='')


#Data cleansing
traindata$Embarked[traindata$Embarked==''] <- "S"
traindata$Age[is.na(traindata$Age)] <- median(traindata$Age,na.rm=T)

colSums(is.na(traindata))
colSums(traindata=='')

#Data anonymization and removing irrelevant features
titanic_data <- traindata %>% select(-c(Cabin, PassengerId, Ticket, Name))

tail(titanic_data)

#Data transformation
for (i in c("Survived","Pclass","Sex","Embarked")){
  titanic_data[,i]=as.factor(titanic_data[,i])
}

tail(titanic_data)

levels(titanic_data$Pclass)
levels(titanic_data$Sex)
levels(titanic_data$Embarked)
titanic_data <- dummy_cols(titanic_data, select_columns = c("Pclass","Sex","Embarked"))
titanic_data <- titanic_data %>% select(-c(Pclass, Sex, Embarked))
tail(titanic_data)
sapply(titanic_data, is.numeric)
#data=titanic_data%>% select(-c(Survived,Fare))
numeric_df=titanic_data%>% select(-c(Fare))
tail(numeric_df)
numeric_df <- apply(numeric_df, 2, as.numeric)
tail(numeric_df)
apply(numeric_df, 2, is.numeric)

#test data
summary(testdata)

colSums(is.na(testdata))
colSums(testdata=='')

sapply(testdata, function(x) length(unique(x)))

unique_x <- unique(testdata)

# Compare lengths to check for duplicates
length(x) != length(unique_x)

testdata$Embarked[testdata$Embarked==""] <- "S"
testdata$Age[is.na(testdata$Age)] <- median(testdata$Age,na.rm=T)

test_data <- testdata %>% select(-c(Cabin, PassengerId, Ticket, Name))

tail(test_data)

for (i in c("Survived","Pclass","Sex","Embarked")){
  test_data[,i]=as.factor(test_data[,i])
}

test_data <- dummy_cols(test_data, select_columns = c("Pclass","Sex","Embarked"))
test_data <- test_data %>% select(-c(Pclass, Sex, Embarked))
tail(test_data)
#sapply(test_data, is.numeric)
#data=titanic_data%>% select(-c(Survived,Fare))
#numeric_df=test_data%>% select(-c(Fare))
#tail(numeric_df)
#numeric_df <- apply(numeric_df, 2, as.numeric)
#tail(numeric_df)
#apply(numeric_df, 2, is.numeric)

cor_data = cor(numeric_df)
corrplot(cor_data, method="circle")

#train <- titanic_data[1:891,]
#test <- titanic_data[892:1309,]
train<-titanic_data
test<-test_data

tail(train)
tail(test)

TitanicModel <- glm(Survived ~.,family=binomial(link='logit'),data=train)

## TitanicModel Summary
summary(TitanicModel)

anova(TitanicModel, test="Chisq")

result <- predict(TitanicModel,newdata=test,type="response")
result <- ifelse(result > 0.5,1,0)

print(result)
print(test$Survived)

result_factor <- factor(result, levels = levels(test$Survived))


accuracy <- mean(result_factor == test$Survived)
print(accuracy)


actual <- factor(test$Survived, levels = c(0, 1))
predicted <- factor(result, levels = c(0, 1))

print(actual)

confusionMatrix(data=predicted, reference=actual)

matrix_data1 <- matrix(c(252, 13, 10, 142), nrow = 2, dimnames = list(Prediction = c(0, 1), Reference = c(0, 1)))
Model_cm <- confusionMatrix(matrix_data1)

plot_data1 <- as.data.frame.table(Model_cm$table)
plot1 <- ggplot(plot_data1, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Model", x = "Reference", y = "Prediction", fill = "Count") +
  geom_text(aes(label = Freq), size = 5)

grid.arrange(plot1, ncol = 1)

saveRDS(TitanicModel, "TitaicModel.RDS")