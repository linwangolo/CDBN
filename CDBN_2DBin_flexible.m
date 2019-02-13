


%------------------------------------------------------------------------------%
%               2D BINARY convolutional Deep Belief Networks (general version)
%------------------------------------------------------------------------------%

function [acc] = CDBN_2DBin_flexible(trainData, trainLabels, testData, testLabels, n_layer, hyparam)

%clear all;

% SET DEMO PARAMETERS 
demo_add_noise = 0;


%% ------------------------------ LOAD DATA --------------------------------- %%

%LOAD MNIST DATA TO TEST THE BINARY CDBN

%realdata = csvread('df_pca.csv',1);

%X = realdata(:,1:48)
%Y = realdata(:,49:50)

%trainData = realdata(1:8000,1:48);
%testData = realdata(8001:9803,1:48);
%trainLabels = realdata(1:8000,49:50);
%testLabels = realdata(8001:9803,49:50);



train_data     = trainData;
train_data     = reshape(train_data', [size(train_data,2),1,1,size(train_data,1)]);
test_data      = testData;
test_data      = reshape(test_data', [size(test_data,2),1,1,size(test_data,1)]);


trainL         = trainLabels;
testL          = testLabels;



% ADD NOISE

if demo_add_noise
    fprintf('------------------- ADD NOISE IN TEST DATA ------------------- \n');
    b          = rand(size(test_data)) > 0.9;
    noised     = test_data;
    rnd        = rand(size(test_data));
    noised(b)  = rnd(b);
    test_data  = noised;
end


%% ------------ INITIALIZE THE PARAMETERS OF THE NETWORK -------------------- %%
% LAYER SETTING


for L=1:n_layer
    layer{L} = default_layer2D();
    layer{L}.inputdata      = train_data;
    if L==1
        layer{L}.n_map_v        = 1;
    else 
        layer{L}.n_map_v        = layer{L-1}.n_map_h;
    end
    %layer{L}.n_map_v        = 1;
    layer{L}.n_map_h        = hyparam(4*n_layer+L);
    layer{L}.s_filter       = [hyparam(5*n_layer+L) 1];
    layer{L}.stride         = [hyparam(8*n_layer+L) hyparam(8*n_layer+L)];  
    layer{L}.s_pool         = [hyparam(6*n_layer+L) 1];
    layer{L}.n_epoch        = hyparam(3*n_layer+L);
    layer{L}.learning_rate  = hyparam(2*n_layer+L);
    layer{L}.sparsity       = hyparam(7*n_layer+L);
    layer{L}.lambda1        = hyparam(L);
    layer{L}.lambda2        = hyparam(n_layer+L);
    layer{L}.whiten         = hyparam(9*n_layer+L);
    layer{L}.type_input     = 'Binary'; % OR 'Gaussian' 'Binary'
end


disp(layer);
   
%% ----------- GO TO 2D CONVOLUTIONAL DEEP BELIEF NETWORKS ------------------ %% 
tic;

[model,layer,err] = cdbn2D(layer);
if isnan(err)
    acc = NaN;
    return
end
save('./model/model_parameter','model','layer');

toc;

%%====================================%
% trainD  = model{1}.output;
% trainD1 = model{2}.output;
train_model = {};
for L=1:n_layer
    train_model{L} = model{L}.output;
end
%%====================================%


%% ------------ TESTDATA FORWARD MODEL WITH THE PARAMETERS ------------------ %%
% FORWARD MODEL OF NETWORKS
H = length(layer);
layer{1}.inputdata = test_data;
fprintf('output the testdata features:>>...\n');

tic;
if H >= 2
    
    % PREPROCESSS INPUTDATA TO BE SUITABLE FOR TRAIN 
    layer{1} = preprocess_train_data2D(layer{1});
    model{1}.output = crbm_forward2D(model{1},layer{1},layer{1}.inputdata);
    
    for k = 2:H
        layer{k}.inputdata = model{k-1}.output;
        layer{k} = preprocess_train_data2D(layer{k});
        model{k}.output = crbm_forward2D(model{k},layer{k},layer{k}.inputdata);
    end
    
else
    
    layer{1} = preprocess_train_data2D(layer{1});
    model{1}.output = crbm_forward2D(model{1},layer{1},layer{1}.inputdata);
end

%%====================================%
% testD  = model{1}.output;
% testD1 = model{2}.output;
test_model = {};
for L=1:n_layer
    test_model{L} = model{L}.output;
end
%%====================================%

toc;

%% ------------------------------- Softmax ---------------------------------- %%

fprintf('train the softmax:>>...\n');

tic;


%%====================================%
% TRANSLATE THE OUTPUT TO ONE VECTOR
trainDa = [];
trainLa = [];
for i= 1:size(train_model{1},4)
	a1 = [];

	for j=1:n_layer
		a1 = [a1; reshape(train_model{j}(:,:,:,i),size(train_model{j},2)*size(train_model{j},1)*size(train_model{j},3),1)];
	end
	
    trainDa = [trainDa,a1];
    trainLa = [trainLa;find(trainL(i,:)==1)];
end



testDa = [];
testLa = [];
for i= 1:size(test_model{1},4)
	b1 = [];

	for j=1:n_layer
    	b1 = [b1; reshape(test_model{j}(:,:,:,i),size(test_model{j},2)*size(test_model{j},1)*size(test_model{j},3),1)];
    end

    testDa = [testDa,b1];
    testLa = [testLa;find(testL(i,:)==1)];
end
%%====================================%




%save('./model/model_out','trainDa','trainLa','testDa','testLa');



% TRAIN THE CLASSIFIER & TEST THE TESTDATA

acc = softmaxExercise(trainDa,trainLa,testDa,testLa);

toc;


end


%% ------------------------------- Figure ----------------------------------- %%

%  POOLING MAPS
%figure(1);
%[r,c,n] = size(model{1}.output(:,:,:,1));
%visWeights(reshape(model{1}.output(:,:,:,1),r*c,n)); colormap gray
%title(sprintf('The first Pooling output'))
%drawnow

% ORIGINAL SAMPLE
%figure(2);
%imagesc(layer{1}.inputdata(:,:,:,1)); colormap gray; axis image; axis off
%title(sprintf('Original Sample'));




