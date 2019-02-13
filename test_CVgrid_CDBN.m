
% **** Example (How to use function) *******
%
% hidlaysize1 = [15 30 70];
% hidlaysize2 = [10 20 50];
% trainopt = {'traingd' 'traingda' 'traingdm' 'traingdx'}; 
% maxepoch = [10 20 40 90];
% transferfunc = {'logsig' 'tansig'};
% bestparameters = gridSearchNN(x_train',y_train',hidlaysize1,...
%                               hidlaysize2,trainopt,maxepoch,transferfunc);

% Main:
%parpool('local',4);
myCluster = parcluster('local');
myCluster.NumWorkers = 18;  % 'Modified' property now TRUE
saveProfile(myCluster);    % 'local' profile now updated,
                           % 'Modified' property now FALSE   


realdata = csvread('df_pca.csv',1);
X = realdata(:,1:48);
Y = realdata(:,49:50);
%realdata = csvread('real_data_0830.csv',1,1);
%X = realdata(:,1:156);
%Y = realdata(:,157:158);

% no n_map_v & type_input
n_layer = 2;
map_r = [5 13 20];
filt_r = [5 9 13];
strid_r = 1;
pool_r = 2;
epoch_r = 2;
lr_r = 0.0001;
spars_r = [0.0001 0.001 0.01];
lamb1_r = [0.5 5 10];
lamb2_r = [0.01 0.1 1];
whit_r = 0;


%n_layer = 2;
%map_r = 5;
%filt_r = 5;
%strid_r = 1;
%pool_r = 2;
%epoch_r = 2;
%lr_r = 0.0001;
%spars_r = 0.0001;
%lamb1_r = 5;
%lamb2_r = 0.01;
%whit_r = 0;



hyparam = containers.Map({'n_map_h', 's_filter', 'stride','s_pool',...
                          'n_epoch','learning_rate','sparsity','lambda1',...
                          'lambda2','whiten'},...
                          { repmat({map_r},1,n_layer),  repmat({filt_r},1,n_layer), repmat({strid_r},1,n_layer), repmat({pool_r},1,n_layer) , ...
                            repmat({epoch_r},1,n_layer),  repmat({lr_r},1,n_layer),repmat({spars_r},1,n_layer) ,  repmat({lamb1_r},1,n_layer), ...
                            repmat({lamb2_r},1,n_layer), repmat({whit_r},1,n_layer) });


tic

[bestparameters, bestaccuracy, allaccuracy] = gridSearchNN(X,Y,hyparam, n_layer);

time = toc;
fprintf('time = %g\nTop accuracy = %0.3f%%\n',time, bestaccuracy*100);

writetable(bestparameters, 'SCM_bestHyparm.txt');
csvwrite('SCM_bestAccuracy.csv',bestaccuracy);
csvwrite('SCM_allAccuracy.csv',allaccuracy);
%parpool close;
%delete(mycluster);
