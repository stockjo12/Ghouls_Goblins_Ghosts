### SET UP ###
#Downloading libraries
library(tidyverse)
library(tidymodels)
library(beepr)

#Bringing in Data
library(vroom)
setwd("~/Library/CloudStorage/OneDrive-BrighamYoungUniversity/STAT 348/Coding/GGG")
train <- vroom("train.csv") |>
  mutate(type = as.factor(type))
test <- vroom("test.csv")

### FEATURE ENGINEERING ###
#Making Recipe
ggg_recipe <- recipe(type ~ ., data = train) |>
  step_rm("id") |>
  step_mutate_at("color", fn = factor) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

### WORK FLOW ###
#Support Vector Machine
#Defining Linear Model
svml_model <- svm_linear(cost = tune()) |> #Target: 0.417
  set_engine("kernlab") |>
  set_mode("classification")

#Creating Workflows
#Linear
svml_wf <- workflow() |>
  add_recipe(ggg_recipe)|>
  add_model(svml_model)

### CROSS VALIDATION ###
#Defining Grids of Values
svml_grid <- grid_regular(cost(range = c(0.1, 10)),
                        levels = 3) #3 for Testing; 5 for Results

#Splitting Data
svml_folds <- vfold_cv(train,
                     v = 5,
                     repeats = 1) #1 for Testing; 3 for Results

#Run Cross Validations
svml_results <- svml_wf |>
  tune_grid(resamples = svml_folds,
            grid = svml_grid,
            metrics = metric_set(accuracy))

#Find Best Tuning Parameters
svml_best <- svml_results |>
  select_best(metric = "accuracy")

#Finalizing Workflow
final_svml_wf <- svml_wf |>
  finalize_workflow(svml_best) |>
  fit(data = train)

### SUBMISSION ###
#Making Predictions
svml_pred <- predict(final_svml_wf, new_data = test, type = "class")

#Formatting Predictions for Kaggle
kaggle_svml <- svml_pred |>
  bind_cols(test) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

#Saving CSV File
vroom_write(x = kaggle_svml, file="./SVML_CV.csv", delim=",")
