library(dplyr)
library(fastDummies)
library(caret)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(grid)
library(tidyr)

traindata = read.csv("train.csv")
testdata = read.csv("test.csv")

head(traindata)
head(testdata)

#train data preprocessing
summary(traindata)

#data visualization
ggplot(traindata, aes(x = factor(Survived))) +
  geom_bar() +
  scale_x_discrete(labels = c("Died", "Survived")) +
  xlab("Survived") +
  ylab("Count") +
  ggtitle("Distribution of Survived")

survivors_df <- subset(traindata, Survived == "1")
print(survivors_df)

survivor_counts = table(survivors_df$Sex)
print(survivor_counts)

survivor_counts_df <- data.frame(Gender = names(survivor_counts), Count = as.numeric(survivor_counts))

ggplot(survivor_counts_df, aes(x = Gender, y = Count, fill = Gender)) +
  geom_bar(stat = "identity") +
  ggtitle("Number of Titanic Survivors by Gender") +
  xlab("Gender") +
  ylab("Number of Survivors")

ggplot(traindata, aes(x = Sex, fill = Sex)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=2, size=4, color="black") +
  ggtitle("Gender Distribution in Titanic Dataset") +
  xlab("Gender") +
  ylab("Count") +
  scale_fill_manual(values = c("#1F77B4", "#FF7F0E"), labels = c("Female", "Male")) +
  theme_classic()

ggplot(survivors_df, aes(x = factor(Pclass))) +
  geom_bar(fill = "steelblue") +
  geom_text(stat='count', aes(label=..count..), vjust=2, size=4, color="black") +
  ggtitle("Survivors by Passenger Class in Titanic Dataset") +
  xlab("Passenger Class") +
  ylab("Count") +
  scale_x_discrete(labels = c("1st Class", "2nd Class", "3rd Class")) +
  theme_classic()

ggplot(traindata, aes(x = factor(Pclass))) +
  geom_bar(fill = "steelblue") +
  geom_text(stat='count', aes(label=..count..), vjust=2, size=4, color="black") +
  ggtitle("Passenger Class Distribution in Titanic Dataset") +
  xlab("Passenger Class") +
  ylab("Count")


ggplot(traindata, aes(x = Age)) +
  geom_histogram(fill = "tomato", color = "white", bins = 30) +
  ggtitle("Age Distribution in Titanic Dataset") +
  xlab("Age") +
  ylab("Count")


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

train<-titanic_data
test<-test_data

tail(train)
tail(test)

TitanicModel <- glm(Survived ~.,family=binomial(link='logit'),data=train)

summary(TitanicModel)

anova(TitanicModel, test="Chisq")

result <- predict(TitanicModel,newdata=test)
print(result)
result <- ifelse(result > 0.5,1,0)

print(result)
print(test$Survived)

sum(is.na(test$Survived))
sum(is.na(result))

missing_index <- which(is.na(result))
if (length(missing_index) > 0) {
  result[missing_index] <- mean(result, na.rm = TRUE)
  result[missing_index] <- ifelse(result > 0.5,1,0)
}

sum(is.na(test$Survived))
sum(is.na(result))

result_factor <- factor(result, levels = c(0, 1))
test_factor <- factor(test$Survived, levels = c(0, 1))

conf_mat = confusionMatrix(data=result_factor, reference=test_factor)
print(conf_mat)

tp <- conf_mat$table[1,1]
tn <- conf_mat$table[2,2]
fp <- conf_mat$table[2,1]
fn <- conf_mat$table[1,2]

cat("True positive:", tp, "\n")
cat("True negative:", tn, "\n")
cat("False positive:", fp, "\n")
cat("False negative:", fn, "\n")

acc = (tp+tn)/(tp+tn+fp+fn)
cat(acc*100,"% \n")

#Crosschecking the accuracy by calculating the mean
accuracy <- mean(result_factor == test_factor)
print(accuracy)

accuracy <- conf_mat$overall[1]
precision <- conf_mat$byClass[1]
recall <- conf_mat$byClass[2]
f1_score <- conf_mat$byClass[3]

cat("Accuracy:", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1 score:", round(f1_score, 3), "\n")

matrix_data1 <- matrix(c(262, 4, 30, 122), nrow = 2, dimnames = list(Prediction = c(0, 1), Reference = c(0, 1)))
Model_cm <- confusionMatrix(matrix_data1)

plot_data1 <- as.data.frame.table(Model_cm$table)
plot1 <- ggplot(plot_data1, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Model", x = "Reference", y = "Prediction", fill = "Count") +
  geom_text(aes(label = Freq), size = 5)

grid.arrange(plot1, ncol = 1)

saveRDS(TitanicModel, "TitaicModel.RDS")