clc;
clear all
close all

load test_sp2015_v14;
load train_sp2015_v14;
train_data=[train_sp2015_v14(1:5000,:); -train_sp2015_v14(10001:15000,:)];
[N,d] = size(train_data);


target=[ones(N/2,1);-ones(N/2,1)];
% pre processing the test and training data

min_train=min(train_data);
max_train=max(train_data);

for i=1:10000
  mat_1(i,:)=train_data(i,:)-min_train(1,:);
  mat_2(i,:)=mat_1(i,:)./(max_train-min_train);
end

inst_mat=sparse(mat_2);
libsvmwrite('svmtrain.txt',target,inst_mat);

[target,inst_mat]=libsvmread('svmtrain.txt');


% ****************************PARAMETER SELECTION***************************
% to compute the best c and gamma values

% parameter _ grid
grid_rep = 5; 
[par_c,gamma_par] = meshgrid(-5:2:15, -15:2:3); 
% grid search and validation of parameters 
cv_acr = zeros(numel(par_c),1); 
d= 2;
count = 0;

% training for 5 subsets of training data
for i=1:5
    count = count+1
    cv_acr(i) = svmtrain(target,inst_mat, ...          
        sprintf('-c %f -g %f -v %d -t %d' , 2^par_c(i), 2^gamma_par(i), grid_rep,d));
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acr); 
%# contour plot of paramter selection 

contour(par_c, gamma_par, reshape(cv_acr,size(par_c))), colorbar
hold on;
text(par_c(idx), gamma_par(idx), sprintf('Acc = %.2f %%',cv_acr(idx)), ...  
    'HorizontalAlign','left', 'VerticalAlign','top') 
hold off 

xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy') 
%# now you can train you model using best_C and best_gamma
best_C = 2^par_c(idx); best_gamma = 2^gamma_par(idx); %# ...

best_cv = 0;
for log2c = -1:3,
  for log2g = -4:1,
    cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(target ,inst_mat, cmd);
    if (cv >= best_cv),
      best_cv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, best_cv);
  end
end

fprintf('best c = %0.6f\n', bestc);
fprintf('best gamma = %0.6f\n', bestg);

fprintf('SVM Classification using rbf kernel\n');
rbf_model = svmtrain(target, inst_mat,'-t 2 -c 8 -g 2 -b 0 -h 0 ');
[rbf_predict_labels]=svmpredict(target,inst_mat,rbf_model);

fprintf('SVM Classification using linear kernel\n');
linear_model = svmtrain(target, inst_mat,'-t 0 -c 8 -g 2 -b 0 -h 0 ');
[linear_predict_labels]=svmpredict(target,inst_mat,linear_model);