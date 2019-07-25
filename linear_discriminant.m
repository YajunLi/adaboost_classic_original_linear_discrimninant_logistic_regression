function [yfit,trained_model]= linear_discriminant(Xtrain,Ytrain, Xtest)

% yfit is prediction, the trained model is saved in trained_model

   Xtrain(~isfinite(Xtrain))=0;
   trained_model = fitcdiscr(Xtrain,Ytrain,'DiscrimType','linear');
   yfit = predict(trained_model,Xtest);
   
end