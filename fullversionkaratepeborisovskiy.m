% Written by Ozan Karatepe, Dmitry Borisovskiy
%% READ ME
% THIS BREAKDOWN OF MODELS FROM TRAINING TO TESTING
% PLEASE RUN - TAKES ESTIMATED 5 MINS TO RUN

%% Objective 
% Compare and evaluate the performance of Naïve Bayes and Random Forest in
% application to binary classification problem – predicting if the 
% participants have heart disease or not. 
% This is a classification task
% Models will be anaylsed on their accuracy and performance
% CSV file is obtained from the UCI Machine Learning Respository 
% Data Set: Cleveland Heart Disease Data Set from UCI Repository. 
% The data set consists of 303 observations, 13 features
% (5 continuous  and 8 categorical) and 1 attribute called ‘target’ which 
% identifies the presents of heart disease. 
% All attributes are numercial
% The dataset will be split into 70% training and 30% will be reserved for
% testing
%% Clear all
clear all; clc; close all;

%% Load data

x = readtable('heart.csv');
%% Data exploration


% The models will be using supervised learning where the targets are
% labelled -  1 = heart disease is present,0 = heart disease not present
% Checking the first 5 rows. 

A = head(x)


%% Shuffle observations - shuffle was added to avoid overfitting and reduce variance

T = x(randperm(size(x,1)),:);

%% Improve readability
% To give a better understanding of the dataset, the coloumn headings are 
% changed to improve readability

T.Properties.VariableNames{'cp'} = 'chest_pain' 
T.Properties.VariableNames{'trestbps'} = 'resting_blood_pressure'
T.Properties.VariableNames{'chol'} = 'cholesterol'
T.Properties.VariableNames{'fbs'} = 'fasting_blood_sugar'
T.Properties.VariableNames{'restecg'} = 'resting_ecg'
T.Properties.VariableNames{'thalach'} = 'max_heart_rate'
T.Properties.VariableNames{'exang'} = 'exercised_induced_angina'
T.Properties.VariableNames{'oldpeak'} = 'ST_depression'
T.Properties.VariableNames{'slope'} = 'peak_exercise_ST_slope'
T.Properties.VariableNames{'ca'} = 'no_major_vessels' 
T.Properties.VariableNames{'thal'} = 'thalassemia' 
%% checking for missing values 
missing = ismissing(T)
%%  To show class balance we build the histogram.
% Shows whether classes are balanced or not
% Plotted as histogram
c1 = categorical(T.target,[1 0], {'Present', 'Not Present'});
h1 = histogram(c1, 'FaceColor', 'b')
ylabel({'No. of patients'});
xlabel({'The  presence of heart disease'});
title({'Figure 1 - Breakdown of the target groups'});
%%  Exploring correlation coefficients between continious predictors.
% Continuous data seperated
continious_data = [T.age, T.resting_blood_pressure, T.cholesterol, T.max_heart_rate, T.ST_depression] 
% find correlation
C = corrcoef(continious_data)
% Plot heatmap
xvalues = {'Age','Rest.BP.','Chol', 'Max.HR', 'ST.dep'};
yvalues = {'Age','Rest.BP.','Chol', 'Max.HR', 'ST.dep'};
hmap = heatmap(xvalues,yvalues,C)
hmap.Title = 'Correlation Coefficients Heatmap For Continuous Predictors';

%%   Identify patients with heart disease 
x = T(T.target == 1, :);
disp([num2str(height(x)) ' Patients with heart disease '])
%%  Identify patients without heart disease 
o = T(T.target == 0, :);
disp([num2str(height(o)) ' Patients without heart disease '])
%% 1.Building histogram to explore the heart disease distribution for age
xage = x.age
oage = o.age
h2 = histogram(xage,'FaceColor', 'red')
hold on
h3 = histogram(oage,'FaceColor', 'blue')
ylabel({'Number Of Observations'});
xlabel({'Age'});
title({'Distribution of Heart Disease by Age'});
legend('With Heart Disease','Without Heart Disease','Position',[0.149702386939454 0.823607522145193 0.307142850703427 0.0869047596341087]);
hold off
%% 2.Building histogram to explore the heart disease distribution for gender
xs = categorical(x.sex,[1 0],{'M','F'})
os = categorical(o.sex,[1 0],{'M.','F.'})
h4 = histogram(xs,'FaceColor', 'red')
hold on
h5 = histogram(os,'FaceColor', 'blue')
ylabel({'Number Of Observations'});
xlabel({'Male (M) / Female (F) with / without Heart Disease (HD)'});
title({'Distribution of Heart Disease by Gender'});
legend('With Heart Disease','Without Heart Disease','Position',[0.580059529796597 0.818627453138026 0.307142850703427 0.0825791833627274]);
hold off

%% Seperation of categorical attributes
heart_sex = categorical(T.sex);
heart_cp = categorical(T.chest_pain);
heart_ecg = categorical(T.resting_ecg);
heart_exc = categorical(T.exercised_induced_angina);
heart_st = categorical(T.peak_exercise_ST_slope);
heart_ves = categorical(T.no_major_vessels);
heart_thal = categorical(T.thalassemia);
heart_target = categorical(T.target);
heart_rbs = categorical(T.fasting_blood_sugar);
% Create table for categorical predictors only
heart_cat = table(heart_sex,heart_cp,heart_ecg,heart_exc,heart_st,heart_ves,heart_thal, heart_rbs, heart_target);

%% Normalisation of continuous attributes
% Zscore normalisation is used
% Avoids any issues with outliners
norm_age = zscore(T.age);
norm_bp = zscore(T.resting_blood_pressure);
norm_chol = zscore(T.cholesterol);
norm_hr = zscore(T.max_heart_rate);
norm_dep = zscore(T.ST_depression);
% Create table for normalised continuous variables only
heart_con = table(norm_age,norm_bp,norm_chol,norm_hr,norm_dep,heart_target);
%% Method 1 - Mixed distribution for all predictors
% Using this method, models will use mixed distribtions:
% Multivariate Multinomial distribution for categorical predictors
% Gaussian/Kernel distribution for continuous predictors

% Create table for categorical and normalised continuous predictors
heart_all = table(heart_sex,heart_cp,heart_ecg,heart_exc,heart_st,heart_ves,heart_thal,heart_rbs,norm_age,norm_bp,norm_chol,norm_hr,norm_dep,heart_target);

% List predictors
predictors = {'heart_sex','heart_cp','heart_ecg','heart_exc','heart_st','heart_ves','heart_thal','heart_rbs','norm_age','norm_bp','norm_chol','norm_hr','norm_dep'};
% Add distributions for each predictor - 
% Multivariate Multinomial distribution / Gaussian Distribution
dis_methods1 = {'mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','normal','normal','normal','normal','normal'};
% Multivariate Multinomial distribution / Kernel Distribution
dis_methods2 = {'mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','kernel','kernel','kernel','kernel','kernel'}; 
% Seperate categorical predictors of Fitcnb
cat_predictors = {'heart_sex','heart_cp','heart_ecg','heart_exc','heart_st','heart_ves','heart_thal','heart_rbs'};
%% Method 2 - Bin continuous data into categories
% Discretise data
% This method is aims to change all attributes into discrete values
% Continuous data are split into 10 bins
% 10 bins was chosen from testing the models to find the optimal number
% Continuous data is now categorical

bin_age = discretize(T.age,10)
bin_bp = discretize(T.resting_blood_pressure,10)
bin_chol = discretize(T.cholesterol,10)
bin_hr = discretize(T.max_heart_rate,10)
bin_dep = discretize(T.ST_depression,10)

% Binned data added to 1 table containing all attributes and target

heart_bin = table(bin_age,bin_bp,bin_chol,bin_hr,bin_dep,heart_sex,heart_cp,heart_ecg,heart_exc,heart_st,heart_ves,heart_thal,heart_rbs,heart_target);

%% Split Data into training and test data
% Holdout method will be implemented with 30% being reserved for testing

%%% Partition of categorical
part_cat = cvpartition(heart_cat.heart_target,'Holdout',0.3);
train_idx_cat = training(part_cat);
test_idx_cat = test(part_cat);

% Training and testing datasets
train_data_cat = heart_cat(train_idx_cat,:);
test_data_cat = heart_cat(test_idx_cat,:);

%%% Partition of continuous

part_con = cvpartition(heart_con.heart_target,'Holdout',0.3);
train_idx_con = training(part_con);
test_idx_con = test(part_con);

% Training and testing datasets
train_data_con = heart_con(train_idx_con,:);
test_data_con = heart_con(test_idx_con,:);


%%% Patition of datasets all being categorical
part_bin = cvpartition(heart_bin.heart_target,'Holdout',0.3);
train_idx_bin = training(part_bin);
test_idx_bin = test(part_bin);

% Training and testing datasets
train_data_bin = heart_bin(train_idx_bin,:);
test_data_bin = heart_bin(test_idx_bin,:);

%%% Partition for complete dataset - categorical and normalised continuous
% data

part_all = cvpartition(heart_all.heart_target,'Holdout',0.3);
train_idx_all = training(part_all);
test_idx_all = test(part_all);

% Training and testing datasets
train_data_all = heart_all(train_idx_all,:);
test_data_all = heart_all(test_idx_all,:);


%% -------------------------- Naive Bayes --------------------------------
%% Train models
tic % add timer
for i = 1:10 % add loop, decided 10 iterations beforehand

% Model 1 - Multivariate Multinomial and Normal distribution with all
% predictors
    nb_mbl1{i} = fitcnb(train_data_all,'heart_target','PredictorNames',predictors,'DistributionNames',dis_methods1);
    train_error1 = resubLoss(nb_mbl1{i})
    nb_mbl1crossval{i} = crossval(nb_mbl1{i}); 
    loss_normal1 = kfoldLoss(nb_mbl1crossval{i});
    k1(i) = loss_normal1;
     
% Model 2 - Multivariate Multinomial and Kernel distribution with all
% predictors  
    nb_mbl2{i} = fitcnb(train_data_all,'heart_target','PredictorNames',predictors,'DistributionNames',dis_methods2);
    train_error2 = resubLoss(nb_mbl2{i})
    nb_mbl2crossval{i} = crossval(nb_mbl2{i}); 
    loss_kernel2 = kfoldLoss(nb_mbl2crossval{i});
    k2(i) = loss_kernel2;

% Model 3 - Multivariate Multinomial and Kernel distribution with all
% predictors and triangle smoothing   
    nb_mbl3{i} = fitcnb(train_data_all,'heart_target','PredictorNames',predictors,'DistributionNames',dis_methods2,'kernel', 'triangle');
    train_error3 = resubLoss(nb_mbl3{i})
    nb_mbl3crossval{i} = crossval(nb_mbl3{i}); 
    loss_kernel3 = kfoldLoss(nb_mbl3crossval{i});
    k3(i) = loss_kernel3;
    
% ----- Using method 3 - continuous predictors are binned    
% Model 4 - Multivariate Multinomial distribution with all predictors
    nb_mbl4{i} = fitcnb(train_data_bin,'heart_target','DistributionNames','mvmn') 
    train_error4 = resubLoss(nb_mbl4{i})
    nb_mbl4crossval{i} = crossval(nb_mbl4{i}); 
    loss_mvmn4 = kfoldLoss(nb_mbl4crossval{i});
    k4(i) = loss_mvmn4;
     
% ----- Using methods 2 - dataset split into subsets
% categorical and continuios

% Model 5 - Normal distribution with continuous predictors only
    nb_mbl5{i} = fitcnb(train_data_con,'heart_target') 
    train_error5 = resubLoss(nb_mbl5{i})
    nb_mbl5crossval{i} = crossval(nb_mbl5{i}); 
    loss_normal5 = kfoldLoss(nb_mbl5crossval{i});
    k5(i) = loss_normal5;
        
% Model 6 - kernel distribution with continuous predictors only
    nb_mbl6{i} = fitcnb(train_data_con,'heart_target', 'DistributionNames','kernel') 
    train_error6 = resubLoss(nb_mbl6{i})
    nb_mbl6crossval{i} = crossval(nb_mbl6{i}); 
    loss_kernel6 = kfoldLoss(nb_mbl6crossval{i});
    k6(i) = loss_kernel6;
       
% Model 7 - Multivariate Multinomial Distribution with categorical
% predictors only
    nb_mbl7{i} = fitcnb(train_data_cat,'heart_target','DistributionNames','mvmn') 
    train_error7 = resubLoss(nb_mbl7{i})
    nb_mbl7crossval{i} = crossval(nb_mbl7{i}); 
    loss_mvmn7 = kfoldLoss(nb_mbl7crossval{i});
    k7(i) = loss_mvmn7;
            
end
toc
%% ---------------------- Cross validation results-------------------------

% Gain the mean cross validatoion errors for each model from 10 iterations
k1 = mean(k1);
k2 = mean(k2);
k3 = mean(k3);
k4 = mean(k4);
k5 = mean(k5);
k6 = mean(k6);
k7 = mean(k7);
% Save cross validation errors and train erros
array = [train_error1,k1; train_error2,k2; train_error3,k3;  train_error4,k4;  train_error5,k5;  train_error6,k6; train_error7,k7];     

% Plot graph to compare models
figure;
bar(array)
legend({'Train Error', 'Validation Error'})
xlabel({'Models'});
ylabel({'Error'});

%% Prior added to Model 1
% In order attempt to improve model 1, the prior is added. 
% 54% of the oberservations have heart disease
% 0.54 prior is added for analysis

% Model 1 - Multivariate Multinomial and Normal distribution with all
% predictors
prior = [0.46,0.54]; 
nb_mbl1 = fitcnb(train_data_all,'heart_target','PredictorNames',predictors,'DistributionNames',dis_methods1, 'Prior', prior);
train_error8 = resubLoss(nb_mbl1)

% Plot to see little improvement using class prior from the dataset.
model1_op = [train_error1;train_error8]
figure;
bar(model1_op)
legend({'Train Error', 'Validation Error'})
xlabel({'Training error'});
ylabel({'Error'});


%% ---------------------------- Test final model ---------------------------
% Model 1 was shown to be the best model.
tic
nb_mbl1 = fitcnb(test_data_all,'heart_target','PredictorNames',predictors,'DistributionNames',dis_methods1, 'Prior', prior);
test_error_nb = loss(nb_mbl1,test_data_all,'heart_target') % Save test error
nb_predict = predict(nb_mbl1,test_data_all) % Save prediction scores
toc

%% --------------------- Naive Bayes Performance -------------------------
% Confusion matrix
% Accuracy, precision, recall and F1 score

% Plot confusion matrix
nb_cm = confusionchart(test_data_all.heart_target,nb_predict)

% Calculate Accuracy
% Accuracy will measure how well the model is at labelling the data
% correctly
% This is an important performance indicator considering the target is well
% balanced.
% Accuracy = (TP+TN)/(TP+FP+FN+TN)
nb_accuracy = sum(diag(nb_cm.NormalizedValues))/(sum(sum(nb_cm.NormalizedValues)));

% Calculate Precision
% Precision will show how the model will label the positive compared to all positive labels available 
% Precision = TP/(TP+FP)
nb_precision = sum(nb_cm.NormalizedValues(1:1))/sum(nb_cm.NormalizedValues(1,:));

% Calculate Recall
% Recall will show the the people who are positive labelled with heart
% disease compared to the number of people with heart disease
% Recall = TP/(TP+FN)
nb_recall = sum(nb_cm.NormalizedValues(1:1))/sum(nb_cm.NormalizedValues(:,1));

% F1-score 
% To find the balance between precision and recall
% harmonic mean(average) of the precision and recall
% F1 Score = 2*(Recall * Precision) / (Recall + Precision)
nb_f1 = 2*(nb_precision*nb_recall)/(nb_recall + nb_precision);

% ROC Curve for model
% ROC curve is a probability curve
% AUC closer to 1 means the classifer is better at distinguishing if a
% patient has heart disease or not.
% The better AUC the better seperability the model has.
target_table = table(test_data_all.heart_target) % Create table for target
test_target = table2array(target_table); % Change table to array


%% -------------------------- Random Forest ------------------------------
% Train Models

% Hyperparameters:
% Number of trees - num_trees
% Min leaf size - 'MinLeafSize'
% Number of parameters - 'NumPredictorsToSample'

% Model 1 - Baseline model set to 500 trees, min leaf size, number of
% paramters left as default
tic
num_trees= 500; % set number of trees to 500

% Use complete dataset, OOB prediction on to plot prediction results
rf_mdl1 = TreeBagger(num_trees,train_data_all,'heart_target','OOBPrediction','on','method','classification'); 
% plot figure - out of bag error reduces as number of trees increase
% 50 trees seen initially as the optimal number of trees
figure('name','Model 1 - Bassline model');
rf_mdl1_oobError = oobError(rf_mdl1);
plot(rf_mdl1_oobError)
xlabel 'Number of trees';
ylabel 'Out-of-bag classification error';
toc
%% ------------------------- Curvature Test--------------------------------
% Plot predictor importance from initial baseline test
rf_mdl1 = TreeBagger(num_trees,train_data_all,'heart_target','PredictorSelection','curvature','OOBPredictorImportance','on','method','classification'); 

figure;
bar(rf_mdl1.OOBPermutedPredictorDeltaError);
title('Curvature Test');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = predictors;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

%% ---------------------------- Optimise models ---------------------------
% Adjust hyperparameters to find optimal model
tic
NumTrees = [50 100 150 200 250]; % Number of trees
MinLeafSize = [2 3]; % Minium number of observations per leaf
NumPredictors = [5 6 7 8 9 10 11 12 13]; % Number of parameters included in model

% Set loop to find best hyperparameters
bestM = 100000;

bestNumTrees = 0;
bestMinLeafSize = 0;
bestNumPredictors= 0;

for j = 1:length(NumTrees)
    for m = 1:length(MinLeafSize)
        for o = 1:length(NumPredictors)    
          
            
              
                    rf_mdl = TreeBagger(NumTrees(j),train_data_all,'heart_target','NumPredictorsToSample',NumPredictors(o),'MinLeafSize',MinLeafSize(m),'OOBPrediction','on','method','classification');  
                    [error] = rf_mdl.oobError;
                    
                    [M,I] = min(error);
                    % Save best hyperparameters
                    if M< bestM
                        bestM=M;
                        bestNumTrees=NumTrees(j);
                        bestMinLeafSize=MinLeafSize(m);
                        bestNumPredictors=NumPredictors(o);
                    end 
                    
               
        end
    end
end 
toc
%% ---------------------------- Test Model --------------------------------
% From optimisation we can discover the best model.
% Hyperparameters are optimised
tic
rf_mdlf = TreeBagger(bestNumTrees,test_data_all,'heart_target','NumPredictorsToSample',bestNumPredictors,'MinLeafSize',bestMinLeafSize,'OOBPrediction','on','OOBPredictorImportance','on','method','classification'); 
test_error_rf = rf_mdlf.oobError;
toc

%% Plot of final model
view(rf_mdlf.Trees{50},'Mode','graph')
%% ----------------------- Random Forest Performance -----------------------  
% Confusion matrix
% Accuracy, precision, recall and F1 score

% Calculate predictions
rf_predict = predict(rf_mdlf,test_data_all);

% Create confusion matrix 
rf_cm = confusionchart(cellstr(string(test_data_all.heart_target)),rf_predict)

% % Calculate Accuracy
% Accuracy will measure how well the model is at labelling the data
% correctly
% This is an important performance indicator considering the target is well
% balanced.
% Accuracy = (TP+TN)/(TP+FP+FN+TN)
rf_accuracy = sum(diag(rf_cm.NormalizedValues))/(sum(sum(rf_cm.NormalizedValues)));

% Calculate Precision
% Precision will show how the model will label the positive compared to all positive labels available 
% Precision = TP/(TP+FP)
rf_precision = sum(rf_cm.NormalizedValues(1:1))/sum(rf_cm.NormalizedValues(1,:));

% Calculate Recall
% Recall will show the the people who are positive labelled with heart
% disease compared to the number of people with heart disease
% Recall = TP/(TP+FN)
rf_recall = sum(rf_cm.NormalizedValues(1:1))/sum(rf_cm.NormalizedValues(:,1));

% F1-score 
% To find the balance between precision and recall
% harmonic mean(average) of the precision and recall
% F1 Score = 2*(Recall * Precision) / (Recall + Precision)
rf_f1 = 2*(rf_precision*rf_recall)/(rf_recall + rf_precision);

%% Predictor Importance for final model

figure
bar(rf_mdlf.OOBPermutedPredictorDeltaError)
title('Predictor Importance - Final Model');
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')
h = gca;
h.XTickLabel = predictors;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';


%% ------------------------ Compare Models ------------------------------
% ROC Curve for models
% ROC curve is a probability curve
% AUC closer to 1 means the classifer is better at distinguishing if a
% patient has heart disease or not.
% The better AUC the better seperability the model has.
target_table = table(test_data_all.heart_target)
test_target = table2array(target_table);
% NB Roc curve - calculate score
[labels,score_nb] = resubPredict(nb_mbl1); % Generate predictions for NB classifer
% Create, X,Y,table for ROC curve and AUC 
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(test_target,score_nb(:,2),1)

% RF calculate score
[labels_rf, score_rf] = oobPredict(rf_mdlf);

% Use score to create, X,Y for ROC curve and AUC
target_table_rf = table(test_data_all.heart_target);
test_target_rf = table2array(target_table_rf);
[Xrf,Yrf,Trf,AUCrf] = perfcurve(test_target_rf,score_rf(:,2),1)

% Plot ROC curve for Naive Bayes and Random Forest models
plot(Xnb,Ynb)
hold on
plot(Xrf,Yrf)
legend('Naive Bayes','Random Forest')
xlabel('False positive rate, 1 - Specificity'); ylabel('True positive rate, Sensitivity');
hold off
% AUC for each model
display(AUCnb) 
display(AUCrf)

