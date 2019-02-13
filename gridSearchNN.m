%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Search  + cross validation funciton				   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%% grid function:

function [out,top_acc,valscores] = gridSearchNN(trainX,trainY,hyparam, n_layer)



  %%-----------test------------%%
  %XTRAIN=trainX(1:1000,:);
  %XTEST=trainX(1001:2000,:);
  %YTRAIN=trainY(1:1000,:);
  %YTEST=trainY(1001:2000,:);

  v = values(hyparam);


    N = 10*n_layer;
   [GRID{1:N}] = ndgrid(v{1}{:},v{2}{:},v{3}{:},v{4}{:},v{5}{:},...
                  v{6}{:},v{7}{:},v{8}{:},v{9}{:},v{10}{:});
    pairs = reshape(cat(N+1,GRID{:}),[],N);
  
    valscores = zeros(size(pairs,1),1);


  %%%%%%%
  %f = fopen('SCM_result.txt', 'w');
  %fprint('Pairs\t\tAccuracy\n');
  parfor i=1:size(pairs,1)
     %fprintf('RUNNING PAIRS %d\n\n', i);
     
     try
     vals = crossval(@(XTRAIN, YTRAIN, XTEST, YTEST)CDBN_2DBin_flexible(XTRAIN, YTRAIN, XTEST, YTEST, n_layer, pairs(i,:)),...
                       trainX, trainY, 'kfold', 3);

    
     fprintf('CV results : %d\n',vals);
     valscores(i) = mean(vals);  

     fprintf('Pairs %d results: %d \n',i, valscores(i));
%     progress = sum(valscores~=0)/length(valscores);
%     fprintf('PROGRESS : %0.3f%%\n', progress*100);          
    
     catch
	vals = nan;
	fprintf('CV results : ERROR \n');
        valscores(i) = mean(vals);

        fprintf('Pairsresults: ERROR \n');

     end


  end
 
  [top_acc ind] = maxk(valscores,10);
  %[max_acc,ind] = max(valscores);
  %out = {pairs(ind,:)};
  out = pairs(ind,:);
  rownames = [];
  for k = keys(hyparam)
    rownames = [rownames, genvarname(repmat(convertCharsToStrings(k),1,n_layer))];
  end
  out = array2table(out,'VariableNames',cellstr(rownames));


end


   
