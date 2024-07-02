#script developed to conduct the basis of my MBA TCC research (Data Science and Analytics)
#manipulation of two main datasets: maneuvers lits, with the dependent variable (swell_obs) and
#waves' forecast. After cleaning both datasets, they are joined together and then used to train
#machine learning classification models (Random Forest and XGBoosting)

library("tidyverse")
library("readxl")
library("lubridate")
library("dplyr")
library("rpart")
library("randomForest")
library("caret")
library("ggplot2")
library("pROC")


#importing the maneuvers dataset (registers since 22/december/2020).
maneuvers <- read_excel("manobras_swell.xlsx")
#importing additional registrations which have been acquired during ships' port stay
maneuvers_extra <- read_excel("swell_2_extra_registers.xlsx")
#concatenating both datasets:
maneuvers <- rbind(maneuvers, maneuvers_extra)

#filling the quay information on a few rows that represent aborted maneuvers 
# (quay came empty on the original dataset)

maneuvers$Cais[maneuvers$Navio == "CMA CGM CAYENNE" & maneuvers$Calado == 6.2 & 
                 maneuvers$Prático == "Bessa" &  maneuvers$`Tipo Man.` == "M"] <- "105"

maneuvers$Cais[maneuvers$Navio == "CHIPOLBROK GALAXY" & maneuvers$Calado == 10.1 & 
                 maneuvers$Prático == "Custódio" &  maneuvers$`Tipo Man.` == "M"] <- "106"

maneuvers$Cais[maneuvers$Navio == "CMM CONTINUITY" & maneuvers$Calado == 2.6 & 
                 maneuvers$Prático == "Pedro" &  maneuvers$`Tipo Man.` == "M"] <- "102"

#dropping rows where the dependent variable is empty:
maneuvers <- maneuvers[complete.cases(maneuvers$`Swell Observado`),]


#Manipulating maneuvers dataset: Simplifying column names; Removing 'port' column;
# Transforming the variable 'man_type' and 'night' to binary format 
# Renaming berths 'P.Ext' and 'P.Int' to, respectively, '202' and '201'
maneuvers <- maneuvers %>% dplyr::rename(ship = 1, ship_type = 2, draft = 3,
                                         loa = 4, beam = 5, dwt = 6, man_type = 7, berth = 8, 
                                         tide_amplitude = 9, port = 10, tide = 11, night = 12,wind_max = 13,
                                         swell_obs = 14, pilot = 15,date_time=16,tide_height=17) %>%
  select(1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17) %>%
  mutate(man_type = recode(man_type,"E" = 0, "M"=0, "S" = 1)) %>%
  mutate(night = recode(night,"Não" = 0, "Sim" = 1)) %>%
  mutate(berth = recode(berth,"P.Ext" = "202", "P.Int" = "201"))

#adjusting all the tide amplitudes to positive values
maneuvers$tide_amplitude <- abs(maneuvers$tide_amplitude)

#adding the month column to capture seasonal patterns:
maneuvers$month <- month(maneuvers$date_time, label = TRUE)

#importing the second dataset (wave data), which has the GFS forecasts for the researched region
#Obs.: swell 2 features included from 08/march/2022.
wave_data <- read_excel("dados_ondas_gfs_wave.xlsx")

#simplifying features' names and dropping the feature "Estação":
wave_data <- wave_data %>% dplyr::rename(estacao = 1, date_time = 2, wave_h = 3,
                                             swell_h = 4, swell2_h = 5, wave_p = 6, 
                                             swell_p = 7, swell2_p = 8, wave_dir = 9,
                                             swell_dir = 10, swell2_dir = 11, sector = 12) %>%
  select(2,3,4,5,6,7,8,9,10,11,12)


# the wave_data has originally a new row every three hours. 
# In the following lines, we add new values every 30 minutes to match the maneuvers observations

interval <- "30 min"

wave_data <- wave_data %>%
  complete(date_time = seq(min(date_time), max(date_time), by = interval)) %>%
  fill(wave_h,swell_h,wave_p,swell2_h,swell_p,swell2_p,wave_dir,swell_dir,swell2_dir,sector)

#join between both datasets, pulling the wave data to the maneuvers (based on the 'date_time').

maneuvers_waves <- inner_join(maneuvers,wave_data,by="date_time")

# transforming angular data, by calculating the sin and cosine values
# 1st component (wave)
# Perform circular-linear transformation
maneuvers_waves$wave_dir_sin <- sin(maneuvers_waves$wave_dir * (pi/180))
maneuvers_waves$wave_dir_cos <- cos(maneuvers_waves$wave_dir * (pi/180))

# 2nd component (swell1)
# Perform circular-linear transformation
maneuvers_waves$swell_dir_sin <- sin(maneuvers_waves$swell_dir * (pi/180))
maneuvers_waves$swell_dir_cos <- cos(maneuvers_waves$swell_dir * (pi/180))

# 3rd component (swell2) 
# Perform circular-linear transformation
maneuvers_waves$swell2_dir_sin <- sin(maneuvers_waves$swell2_dir * (pi/180))
maneuvers_waves$swell2_dir_cos <- cos(maneuvers_waves$swell2_dir * (pi/180))

#defining function to manipulate the tidal information, that came in text format.
#the function extracts sub-strings with low and high tide time, and the time of the maneuver.
# Based on those pieces of information, it identifies if the maneuver happened
# during the high tide, low tide, rising tide or ebb tide.
# To be considered as high or low tide, we admit times happening between 1h30m before and 1h30m after
# the high or low tide time.

extract_tide_info <- function(feature) {
  #manipulating the three 'time' registers:
  time1<-substr(feature, 4, 4 + 5 - 1)
  time2<-substr(feature, 29, 29 + 5 - 1)
  time3<-substr(feature, 63, 63 + 5 - 1)
  
  time1<-strptime(time1, format = "%H:%M")
  time2<-strptime(time2, format = "%H:%M")
  time3<-strptime(time3, format = "%H:%M")
  
  time1<-time1$hour + time1$min / 60
  time2<-time2$hour + time2$min / 60
  time3<-time3$hour + time3$min / 60
  
  if ((is.na(time1))|(is.na(time2))|is.na(time3)){
    return(NA)
  } else {
    
    #checking wether the high tide or low tide happens first, which identifies if it is 
    #either rising or ebb tide:
    if(substr(feature, 18, 18 + 2 - 1)=="BM"){
      status_tide<-"rising"
    } else {
      status_tide<-"ebb"
    }
    
    #verifying differences between time2 and time1, and between time3 and time2, 
    #to identify if the maneuver happened during the slack tide 
    #if time2<time1, or time3<time2, we are having times in different days, 
    #so we need to add 24.
    
    if(time2<time1){
      dif1<-time2+24-time1
    } else {
      dif1<-time2-time1
    }
    
    if(time3<time2){
      dif2<-time3+24-time2
    } else {
      dif2<-time3-time2
    }
    
    #veryfing tide status, if high tide, low tide, rising tide or ebb tide:
    
    if((status_tide=="rising")&(dif1<1.5)){
      tide<-"low"
    } else if ((status_tide=="rising")&(dif2<1.5)){
      tide<-"high"
    } else if ((status_tide=="rising")){
      tide<-"rising"
    } else if ((status_tide=="ebb")&(dif1<1.5)){
      tide<-"high"
    } else if ((status_tide=="ebb")&(dif2<1.5)){
      tide<-"low"
    } else if ((status_tide=="ebb")){
      tide<-"ebb"
    }
    
    return(tide)
  }
}

#applying the above function to add a new feature called tide_phase
maneuvers_waves$tide_phase <- sapply(maneuvers_waves$tide, extract_tide_info)

#transforming several features to factor type:
maneuvers_waves$berth<-factor(maneuvers_waves$berth)
maneuvers_waves$man_type<-factor(maneuvers_waves$man_type)
maneuvers_waves$night<-factor(maneuvers_waves$night)
maneuvers_waves$swell_obs<-factor(maneuvers_waves$swell_obs)
maneuvers_waves$month<-factor(maneuvers_waves$month)
maneuvers_waves$tide_phase<-factor(maneuvers_waves$tide_phase)
maneuvers_waves$sector<-factor(maneuvers_waves$sector)
maneuvers_waves$pilot<-factor(maneuvers_waves$pilot)
maneuvers_waves$ship_type<-factor(maneuvers_waves$ship_type)


#filtering the main dataset based on the dependent variable, to perform EDA

maneuvers_zero <- subset(maneuvers_waves, swell_obs == 0)
maneuvers_01 <- subset(maneuvers_waves, swell_obs == 1)
maneuvers_02 <- subset(maneuvers_waves, swell_obs == 2)




####################################################
# Splitting the dataset for training and testing purposes #
####################################################

#assign random values (1 or 2) for each row, where 1=train;2=validation;3=test
set.seed(1234567)
split_data <- sample(c('train', 'test'),
                    size = nrow(maneuvers_waves),
                    replace = TRUE,
                    prob=c(0.8, 0.2))


# Generating the train, validation e test datasets
train    <-maneuvers_waves[split_data == 'train',]

test     <- maneuvers_waves[split_data == 'test',]

#########################################
# Random Forest                      #
#########################################

#dropping rows with NaN values:
train<-na.omit(train)
test <- na.omit(test)

set.seed(1234567)
rf <- randomForest::randomForest(
  swell_obs~draft+loa+beam+dwt+man_type+berth+tide_amplitude+night+
    tide_height+month+wave_h+swell_h+wave_p+swell_p+sector+
    wave_dir_sin+wave_dir_cos+swell_dir_sin+swell_dir_cos+tide_phase,
  data = train,
  method="class",
  ntree = 500
)

#predction and accuracy calculation using the RandomForest model:
pRF_train <- predict(rf, train,type="class")
pRF_test  <- predict(rf, test,type="class")

acc_test_RF <- sum(pRF_test == test$swell_obs)/nrow(test)
acc_test_RF #0.825

#####################################
# Random-forest plus gridsearch      #
#####################################

# Defining hyperparameters for the grid search
hyperparameters <- expand.grid(mtry = c(4,5, 6, 7, 8,9,10,11,12,13,14))

# Defining control function to cross validation:
ctrl <- trainControl(method = "cv", # CV -> "k-fold cross validation"
                     number = 5)  # 5 is the number of "folds"

# Running the grid search with cross validation:
set.seed(1234567)
gridsearch_kfold <- train(swell_obs~draft+loa+beam+dwt+man_type+berth+tide_amplitude+night+
                            tide_height+month+wave_h+swell_h+wave_p+swell_p+sector+
                            wave_dir_sin+wave_dir_cos+swell_dir_sin+swell_dir_cos+tide_phase, 
                          data = train, 
                          method = "rf", 
                          trControl = ctrl, 
                          tuneGrid = hyperparameters)

print(gridsearch_kfold)
plot(gridsearch_kfold)
#optimal model with mtry = 9

ntrees <- gridsearch_kfold$finalModel$ntree

#applying the RF to the test dataset:
p_rftunned <- predict(gridsearch_kfold, test,type="raw")
acc_test_RF_tun <- sum(p_rftunned == test$swell_obs)/nrow(test)
acc_test_RF_tun #0.85

# Generating the confusion matrix and associated statistics
cm <- caret::confusionMatrix(p_rftunned, test$swell_obs)
cm 
#class 2 sensitivity (no threshold) = 0.31
# class 2 specificity = 0.99
# balanced accuracy for class 2 = 0.65

#considering that the swell_obs class "2" is the most crictical for out research,
#and considering its very low occurrence rate, it was stablished a lower threshold
#to classify a given observation as swell_obs class 2. If the probability of classification
#for the class "2" is higher than 15%, this class will be chosen to classift the given row.

# Predict class probabilities on the test set
probs_rf <- predict(gridsearch_kfold, test, type = "prob")

predicted_details_rf <- cbind(p_rftunned, test$swell_obs, probs_rf)

#Set the threshold for the class "2"
threshold <- 0.145

#Adjust predictions based on the threshold
predicted_classes_rf <- apply(probs_rf, 1, function(x) {
  if (x["2"] > threshold) {
    return("2")
  } else {
    return(names(which.max(x)))
  }
})

# Convert to factor to match the levels of the actual classes
predicted_classes_rf <- factor(predicted_classes_rf, levels = levels(test$swell_obs))

acc_test_adjusted <- sum(predicted_classes_rf == test$swell_obs)/nrow(test)
acc_test_adjusted 

cm2 <- confusionMatrix(predicted_classes_rf, test$swell_obs)
cm2 


importance_values <- varImp(gridsearch_kfold, scale = FALSE)
print(importance_values)

# Extract variable importances
variable_importances <- varImp(gridsearch_kfold)

# Convert to data frame
importance_df <- as.data.frame(variable_importances$importance)

# Add variable names as a column
importance_df$Variable <- rownames(importance_df)

# Sort by importance
importance_df <- importance_df[order(importance_df$Overall, decreasing = TRUE),]

# Select the top 17 variables
top_17_importance_df <- importance_df[1:17,]

# Create the barplot (Top 17 Variables Importance from Random Forest)
ggplot(top_17_importance_df, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Flip coordinates for better readability
  xlab("") + 
  ylab("Importance") + 
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 14), # Adjust x-axis label fontsize
    axis.title.y = element_text(size = 14), # Adjust y-axis label fontsize
    axis.text.y = element_text(size = 12),   # Adjust variable names fontsize
    axis.text.x = element_text(size = 12) # Adjust importance values fontsize
  )

#AUC-ROC Curve (OVR - One-vs-rest - approach)

# Calculate the ROC curve for each class
roc_list <- list()
auc_list <- numeric(length(levels(test$swell_obs)))

for (class in levels(test$swell_obs)) {
  roc_curve <- roc(test$swell_obs == class, probs_rf[[class]],legacy.axes=TRUE)
  auc_list[class == levels(test$swell_obs)] <- auc(roc_curve)
  roc_list[[class]] <- roc_curve
}


#the following reference helped me to build the ROC plot script: 
# https://rdrr.io/cran/pROC/man/plot.roc.html

# Open a new plotting window with specific dimensions
par(pty = "s", mfrow = c(1, 1),mar = c(4, 4, 2, 1)) 
plot(roc_list[['0']], col = "black", lwd = 2, grid=TRUE, 
     legacy.axes=TRUE,xlim=c(1, 0),ylim=c(0, 1),
     xlab="1 - Specificity", main = "Multi-Class ROC Curves (Random Forest)")
plot(roc_list[['1']], col = "blue", lwd = 2, grid=TRUE, 
     legacy.axes=TRUE,xlim=c(1, 0),ylim=c(0, 1),
     xlab="1 - Specificity",add=TRUE)
plot(roc_list[['2']], col = "red", lwd = 2, grid=TRUE, 
     legacy.axes=TRUE,xlim=c(1, 0),ylim=c(0, 1),
     xlab="1 - Specificity",add=TRUE)
# Add legend
legend("bottomright", 
       legend = c('Level 0 vs Rest (AUC = 0.93)', 'Level 1 vs Rest (AUC = 0.91)', 
                  'Level 2 vs Rest (AUC = 0.94)','Random Classifier'),
       col = c("black", "blue", "red","gray"),
       lwd = 2,cex = 0.80)


# Calculate the average AUC
average_auc <- mean(auc_list)
print(paste("Average AUC:", average_auc)) # 0.928

#####################################
# XGBoosting      #
#####################################

control_xgb <- caret::trainControl(
  "cv",
  number = 5,
  summaryFunction = defaultSummary, 
  classProbs = TRUE 
)

gridsearch_xgb <- expand.grid(
  nrounds = c(150,250),
  max_depth = c(5, 7),
  gamma = c(0),
  eta = c(0.01, 0.1),
  colsample_bytree = c(0.6,0.8),
  min_child_weight = c(1),
  subsample = c(0.75, 1)
)

# caret::train did not work with "0", "1" e "2", so a quick adjustment was applied:

train <- train %>% 
  mutate(swell_obs = recode(swell_obs,"0" = "zero", "1"="one", "2" = "two"))
test <- test %>% 
  mutate(swell_obs = recode(swell_obs,"0" = "zero", "1"="one", "2" = "two"))

set.seed(1234567)
modelo_xgb <- caret::train(
  swell_obs~draft+loa+beam+dwt+man_type+berth+tide_amplitude+night+
    tide_height+month+wave_h+swell_h+wave_p+swell_p+sector+
    wave_dir_sin+wave_dir_cos+swell_dir_sin+swell_dir_cos+tide_phase, 
  data = train, 
  method = "xgbTree",
  trControl = control_xgb,
  tuneGrid = gridsearch_xgb,
  verbosity = 0)

modelo_xgb

# Extract the number of trees (boosting rounds) used in the final model
nrounds <- modelo_xgb$bestTune$nrounds

# Print the number of trees
print(nrounds)

class_xgb_train <- predict(modelo_xgb, train)
class_xgb_test <- predict(modelo_xgb, test)

acc_test <- sum(class_xgb_test == test$swell_obs)/nrow(test)
acc_test


cm3 <- confusionMatrix(class_xgb_test, test$swell_obs)
cm3 



predicted_probs <- predict(modelo_xgb, newdata = test, type = "prob")

predicted_details <- cbind(class_xgb_test, test$swell_obs, predicted_probs)

# Define a threshold for the class ('two')
threshold <- 0.064

# Custom function to apply the decision rule
get_predicted_classes <- function(prob_matrix, threshold) {
  predicted_classes <- rep("zero", nrow(prob_matrix))  # Initialize with default class ('zero')
  
  # Check if probability of class 'two' is greater than the threshold
  for (i in 1:nrow(prob_matrix)) {
    if (prob_matrix[i, "two"] > threshold) {
      predicted_classes[i] <- "two"  # Assign 'two' if its probability is above threshold
    } else {
      # Assign the class with the maximum probability if 'two' is not above threshold
      predicted_classes[i] <- colnames(prob_matrix)[which.max(prob_matrix[i, ])]
    }
  }
  
  return(predicted_classes)
}

# Get predicted classes based on the custom decision rule
predicted_classes <- get_predicted_classes(predicted_probs, threshold)
predicted_classes_factor <- factor(predicted_classes, levels = levels(test$swell_obs))

acc_test_adjusted <- sum(predicted_classes_factor == test$swell_obs)/nrow(test)
acc_test_adjusted 

# Create a confusion matrix to evaluate performance
cm4 <- confusionMatrix(predicted_classes_factor, test$swell_obs)
cm4 


#train <- train %>% 
#  mutate(swell_obs = recode(swell_obs,"zero" = "0", "one"="1", "two" = "2"))
#test <- test %>% 
#  mutate(swell_obs = recode(swell_obs,"zero" = "0", "one"="1", "two" = "2"))

