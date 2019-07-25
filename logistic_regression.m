function [yfit,trained_model]= logistic_regression(Xtrain,Ytrain, Xtest)

% yfit is prediction, the trained model is saved in trained_model

   Xtrain(~isfinite(Xtrain))=0;
   trained_model = fitclinear(Xtrain,Ytrain,'Learner','logistic');
   yfit = predict(trained_model,Xtest);
   
end