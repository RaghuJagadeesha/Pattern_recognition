clc;
clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Load the training data ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load train_sp2015_v14;

input = train_sp2015_v14';

train_w1 = input(:,1:5000);
train_w2 = input(:,5001:10000);
train_w3 = input(:,10001:15000);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** To check for Gaussian distribution ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%hist(train_w1);
% hist(train_w2);
%hist(train_w3);

class = 3;
[fs,N] = size(input);
N_perclass = N/class;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Assume apriori probability values for each class ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p_w1 = 1/class;
p_w2 = 1/class;
p_w3 = 1/class;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Compute mean for each class ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c_mean_w1 = mean(train_w1')';
c_mean_w2 = mean(train_w2')';
c_mean_w3 = mean(train_w3')';


for j = 1:N_perclass
    x_mean_w1(:,j) = train_w1(:,j)- c_mean_w1;
    x_mean_w2(:,j) = train_w2(:,j)- c_mean_w2;
    x_mean_w3(:,j) = train_w3(:,j)- c_mean_w3;        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Compute the covariance matrix ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
var_w1 = x_mean_w1*x_mean_w1'./N_perclass;
var_w2 = x_mean_w2*x_mean_w2'./N_perclass;
var_w3 = x_mean_w3*x_mean_w3'./N_perclass;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Verify discrimination function using the training data ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:N
    train_data1 = input(:,i)- c_mean_w1;
    train_data2 = input(:,i)- c_mean_w2;
    train_data3 = input(:,i)- c_mean_w3;
    
    disc_w(i,1) = ((-1/2)*(train_data1)'*inv(var_w1)*(train_data1))-((1/2)*log(det(var_w1)))+(log(p_w1))-((fs/2)*log(2*pi));
    disc_w(i,2) = ((-1/2)*(train_data2)'*inv(var_w2)*(train_data2))-((1/2)*log(det(var_w2)))+(log(p_w2))-((fs/2)*log(2*pi));
    disc_w(i,3) = ((-1/2)*(train_data3)'*inv(var_w3)*(train_data3))-((1/2)*log(det(var_w3)))+(log(p_w3))-((fs/2)*log(2*pi));
    
end
output_training = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Classify the training data based on the discrimination function output ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:N
    if(disc_w(i,1)>disc_w(i,2))
        if(disc_w(i,1)>disc_w(i,3))
            output_training(i)=1;
        else
            output_training(i)=3;
        end
    else
        if(disc_w(i,2)>disc_w(i,3))
            output_training(i)=2;
        else
            output_training(i)=3;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Confusion matrix computation ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conf_matrix = zeros(3,3);
count = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ***** Calculate probability of error of the discriminant function ******* %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:5000
    if(output_training(i)==1)
        conf_matrix(1,1)= conf_matrix(1,1)+1;
    elseif(output_training(i)==2)
        conf_matrix(1,2) = conf_matrix(1,2)+1;
    else
        conf_matrix(1,3) = conf_matrix(1,3)+1;
    end
end
error_class_w1 = conf_matrix(1,2)+ conf_matrix(1,3);
for i = 5001:10000
    if(output_training(i)==2)
        conf_matrix(2,2)= conf_matrix(2,2)+1;
    elseif(output_training(i)==1)
        conf_matrix(2,1) = conf_matrix(2,1)+1;
    else
        conf_matrix(2,3) = conf_matrix(2,3)+1;
    end
end
error_class_w2 = conf_matrix(2,1)+ conf_matrix(2,3);
for i = 10001:15000
    if(output_training(i)==3)
        conf_matrix(3,3)= conf_matrix(3,3)+1;
    elseif(output_training(i)==2)
        conf_matrix(3,2) = conf_matrix(3,2)+1;
    else
        conf_matrix(3,1) = conf_matrix(3,1)+1;
    end
end
error_class_w3 = conf_matrix(3,2)+ conf_matrix(3,1);


p_error_w1 = error_class_w1/N_perclass;
p_error_w2 = error_class_w2/N_perclass;
p_error_w3 = error_class_w3/N_perclass;

p_tot_error = (error_class_w1 + error_class_w2 + error_class_w3)/N;

fprintf('Probability of Error : %0.4f \n \n', p_tot_error); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ****** Load the test data ****** %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load test_sp2015_v14;

test_set = test_sp2015_v14';

[fs_t,N_t] = size(test_set);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ****** Calculate the discriminant function ****** %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N_t
    test_data1 = test_set(:,i)-c_mean_w1;
    test_data2 = test_set(:,i)-c_mean_w2;
    test_data3 = test_set(:,i)-c_mean_w3;
    
    disc_test_w(i,1)= ((-1/2)*(test_data1)'*inv(var_w1)*(test_data1))-((1/2)*log(det(var_w1)))+(log(p_w1))-((fs_t/2)*log(2*pi));
    disc_test_w(i,2)= ((-1/2)*(test_data2)'*inv(var_w2)*(test_data2))-((1/2)*log(det(var_w2)))+(log(p_w2))-((fs_t/2)*log(2*pi));
    disc_test_w(i,3)= ((-1/2)*(test_data3)'*inv(var_w3)*(test_data3))-((1/2)*log(det(var_w3)))+(log(p_w3))-((fs_t/2)*log(2*pi));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ****** Classify the test data based on Discriminant function output ****** %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N_t
    if(disc_test_w(i,1)>disc_test_w(i,2))
        if(disc_test_w(i,1)>disc_test_w(i,3))
            output_test(i)=1;
        else
            output_test(i)=3;
        end
    else
        if(disc_test_w(i,2)>disc_test_w(i,3))
            output_test(i)=2;
        else
            output_test(i)=3;
        end
    end
end


fpt1 = fopen('training_classified.txt','w');
fpt2 = fopen('test_classified.txt','w');
for i = 1:N_t
    fprintf(fpt1,'%d\n',output_training(i));
    fprintf(fpt2,'%d\n',output_test(i));
end
fpt3 = fopen('confusion_matrix.txt','w');
for i = 1:3
    for j=1:3
        fprintf(fpt3,'%d ',conf_matrix(i,j));
    end
    fprintf(fpt3,'\n');
end
test_output = output_test';
a = [];
i=1;
while i<=15000
a(i) = 2;
a(i+1) = 3;
a(i+2) = 1;
a(i+3) = 3;
a(i+4) = 1;
a(i+5) =2;
i = i+6;
end
b= a';
fclose(fpt1);
fclose(fpt2);
fclose(fpt3);