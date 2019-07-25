function [ada_train, ada_test, eps, alpha]= adaboost(Xtrain,Ytrain, Xtest)
% AdaBoost function 
% (X_train input: training set)
% (Y_train target)
% (Xtest input: testing set)
% (ada_train-> label: training set)
% (ada_test-> label: testing set)
% eps: sum weight of misprediction
% alpha: amount of say 

% Choosen Weak classifiers:
% 1. linear discriminant
% 2. linear discriminant
% 3. Logistic Regression

N=size(Xtrain,1);
a=[Xtrain Ytrain];

D=(1/N)*ones(N,1);
Dt=[]; h_=[];

Classifiers=3;
eps=zeros(Classifiers,1);
va = 1:length(D);

for T=1:Classifiers
    
    for i=1:length(D)
        
        p_min=min(D);
        p_max=max(D);
        p = (p_max-p_min)*rand(1) + p_min;
        
        if D(i)>=p
            Dt = [Dt;a(i,:)];
        else  
            t = rand_gen(D,va);
            Dt = [Dt;a(t,:)];
        end
    end

    X=Dt(:,1:end-1);
    Y=Dt(:,end);
    
    if T==1
        logis1_in=fitclinear(X,Y,'Learner','logistic');
        linear_out=predict(logis1_in, X);
        h=linear_out;
    end

    if T==2
        logis2_in=fitclinear(X,Y,'Learner','logistic');
        linear_out=predict(logis2_in, X);
        h=linear_out;
    end
    
   if T==3
       % linear discriminant  0.653
       lineardis_in=fitcdiscr(X,Y,'discrimType', 'linear');
       lineardis_out=predict(lineardis_in, X);
       h=lineardis_out;
   end

    Dt = [];
    h_=[h_ h];

    % weighted error
    for i=1:length(Y)
        if (h_(i,T)~=Y(i))
            eps(T)=eps(T)+D(i,:); 
        end  
    end
    
    % Hypothesis weight
    alpha(T)=0.5*log((1-eps(T))/eps(T));
    
    % Update weights
    
    D=D.*exp((-1).*Y.*alpha(T).*h);
    D=D./sum(D);
   
end

% final vote for train (in sample validate)
H(:,1)=predict(logis1_in, Xtrain);
H(:,2)=predict(logis2_in, Xtrain);
H(:,3)=predict(lineardis_in, Xtrain);
ada_train(:,1)=sign(H*alpha');

% for test set
Htest(:,1)=predict(logis1_in, Xtest);
Htest(:,2)=predict(logis2_in, Xtest);
Htest(:,3)=predict(lineardis_in, Xtest);
ada_test(:,1)=sign(Htest*alpha');
end
