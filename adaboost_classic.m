function [estimateclasstotal,model]=adaboost(mode,datafeatures,dataclass_or_model,itt)
% This function AdaBoost, consist of two parts: 
% a simpel weak classifier and a boosting part:
% 1. The weak classifier tries to find the best treshold in one of the data
% dimensions to separate the data into two classes -1 and 1
% 2. The boosting part calls the clasifier iteratively, after every classification
% step it changes the weights of miss-classified examples. 
% It creates a cascade of "weak classifiers" which behaves like a "strong classifier"

% 1. Training mode:
%    [estimateclass,model]=adaboost('train',datafeatures,dataclass,itt)
% 2. Apply mode:
%    estimateclass=adaboost('apply',datafeatures,model)
% 
%  inputs/outputs:
%    datafeatures : An Array with 'size' number of samples and x number of features
%    dataclass : An array of  -1 or 1
%    itt : The number of training iterations
%    model : A struct with the cascade of weak-classifiers
%    estimateclass : forcast
%
%  Function is written by D.Kroon University of Twente (August 2010)

switch(mode)
    case 'train'
        % Train the adaboost model
        
        % Set the data class 
        dataclass=dataclass_or_model;
        model=struct;
        
        % the initial weights 
        D=ones(length(dataclass),1)/length(dataclass);
        
        % contain the results of the weak classifiers weighted by their alpha
        estimateclasssum=zeros(size(dataclass));
        
        % Calculate the min and max of all data features
        boundary=[min(datafeatures,[],1) max(datafeatures,[],1)];
        
        % model training
        for t=1:itt
            % Find the best treshold to separate the data in two classes
            [estimateclass,err,h] = WeightedThresholdClassifier(datafeatures,dataclass,D);

            % amount of say
            alpha=1/2 * log((1-err)/max(err,eps));
            
            % Store the model parameters
            model(t).alpha = alpha;
            model(t).dimension=h.dimension;
            model(t).threshold=h.threshold;
            model(t).direction=h.direction;
            model(t).boundary = boundary;
            
            % update the weight
            D = D.* exp(-model(t).alpha.*dataclass.*estimateclass);
            D = D./sum(D);
            
            % Calculate the error
            estimateclasssum=estimateclasssum +estimateclass*model(t).alpha;
            estimateclasstotal=sign(estimateclasssum);
            model(t).error=sum(estimateclasstotal~=dataclass)/length(dataclass);
            if(model(t).error==0), break; end
        end
        
    case 'apply' 
        % Apply Model on the test data
        model=dataclass_or_model;
        
        % Limit data features to the boundaries
        if(length(model)>1)
            minb=model(1).boundary(1:end/2);
            maxb=model(1).boundary(end/2+1:end);
            datafeatures=bsxfun(@min,datafeatures,maxb);
            datafeatures=bsxfun(@max,datafeatures,minb);
        end
    
        % ensemble all methods to predict
        estimateclasssum=zeros(size(datafeatures,1),1);
        for t=1:length(model)
            estimateclasssum=estimateclasssum+model(t).alpha*ApplyClassTreshold(model(t), datafeatures);
        end
        
        estimateclasstotal=sign(estimateclasssum);
        
    otherwise
        error('adaboost:inputs','unknown mode');
end


function [estimateclass,err,h] = WeightedThresholdClassifier(datafeatures,dataclass,dataweight)

% selects the dimension and treshold which divides the 
% data into two class with the smallest error.

% Number of treshold steps
ntre=2e5;

% Split the data in two classes 1 and -1
r1=datafeatures(dataclass<0,:); w1=dataweight(dataclass<0);
r2=datafeatures(dataclass>0,:); w2=dataweight(dataclass>0);

% Calculate the min and max for every dimension
minr=min(datafeatures,[],1)-1e-10; maxr=max(datafeatures,[],1)+1e-10;

% Make a weighted histogram of the two classes
p2c= ceil((bsxfun(@rdivide,bsxfun(@minus,r2,minr),(maxr-minr)))*(ntre-1)+1+1e-9);   p2c(p2c>ntre)=ntre;
p1f=floor((bsxfun(@rdivide,bsxfun(@minus,r1,minr),(maxr-minr)))*(ntre-1)+1-1e-9);  p1f(p1f<1)=1;
ndims=size(datafeatures,2);
i1=repmat(1:ndims,size(p1f,1),1);  i2=repmat(1:ndims,size(p2c,1),1);

h1f=accumarray([p1f(:) i1(:)],repmat(w1(:),ndims,1),[ntre ndims],[],0);
h2c=accumarray([p2c(:) i2(:)],repmat(w2(:),ndims,1),[ntre ndims],[],0);

% This function calculates the error for every possible treshold value and dimension
h2ic=cumsum(h2c,1);
h1rf=cumsum(h1f(end:-1:1,:),1); h1rf=h1rf(end:-1:1,:);
e1a=h1rf+h2ic;
e2a=sum(dataweight)-e1a;

% We want the treshold value and dimension with the minimum error
[err1a,ind1a]=min(e1a,[],1);  dim1a=(1:ndims); dir1a=ones(1,ndims);
[err2a,ind2a]=min(e2a,[],1);  dim2a=(1:ndims); dir2a=-ones(1,ndims);
A=[err1a(:),dim1a(:),dir1a(:),ind1a(:);err2a(:),dim2a(:),dir2a(:),ind2a(:)];
[err,i]=min(A(:,1)); dim=A(i,2); dir=A(i,3); ind=A(i,4);
thresholds = linspace(minr(dim),maxr(dim),ntre);
thr=thresholds(ind);

% Apply the new treshold
h.dimension = dim; 
h.threshold = thr; 
h.direction = dir;
estimateclass=ApplyClassTreshold(h,datafeatures);

function y = ApplyClassTreshold(h, x)

% Draw a line to separate data into 2 classes
if(h.direction == 1)
    y =  double(x(:,h.dimension) >= h.threshold);
else
    y =  double(x(:,h.dimension) < h.threshold);
end
y(y==0) = -1;


