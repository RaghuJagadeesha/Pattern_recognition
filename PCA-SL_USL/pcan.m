function [tr_tot_p_err,conf_mat_train,tst_tot_p_err,conf_mat_tst] = pcan(train_set,test_set)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[d,N] = size(train_set);
mean_vect = mean(train_set,2);
for i =1:N
    x_mean(:,i) = train_set(:,i)-mean_vect;
end
covar_mat = x_mean*x_mean'./N;

[V,E,R] = svd(covar_mat);
eig_val = [];
eig_vect = [];
for i=1:2
   eig_val(:,i) = E(:,i); 
   eig_vect(:,i) = V(:,i);
end

new_train_set = (train_set'*eig_vect)';

n_train_w1 = new_train_set(:,1:5000);
n_train_w2 = new_train_set(:,5001:10000);
n_train_w3 = new_train_set(:,10001:15000);

c_mean_w1 = mean(n_train_w1,2);
c_mean_w2 = mean(n_train_w2,2);
c_mean_w3 = mean(n_train_w3,2);

for i=1:5000
    x_mean_w1(:,i) = n_train_w1(:,i) - c_mean_w1;
    x_mean_w2(:,i) = n_train_w2(:,i) - c_mean_w2;
    x_mean_w3(:,i) = n_train_w3(:,i) - c_mean_w3;
end

[d_perclass,N_perclass] = size(n_train_w1);

covar_w1 = x_mean_w1*x_mean_w1'./N_perclass;
covar_w2 = x_mean_w2*x_mean_w2'./N_perclass;
covar_w3 = x_mean_w3*x_mean_w3'./N_perclass;

p_w1 = 1/3;
p_w2 = 1/3;
p_w3 = 1/3;

for i = 1:N
    data_w1 = new_train_set(:,i) - c_mean_w1;
    data_w2 = new_train_set(:,i) - c_mean_w2;
    data_w3 = new_train_set(:,i) - c_mean_w3;

    desc_w(i,1) = ((-1/2)*(data_w1)'*inv(covar_w1)*(data_w1))-((1/2)*(log(det(covar_w1))))+(log(p_w1))-((d_perclass/2)*log(2*pi));
    desc_w(i,2) = ((-1/2)*(data_w2)'*inv(covar_w2)*(data_w2))-((1/2)*(log(det(covar_w2))))+(log(p_w2))-((d_perclass/2)*log(2*pi));
    desc_w(i,3) = ((-1/2)*(data_w3)'*inv(covar_w3)*(data_w3))-((1/2)*(log(det(covar_w3))))+(log(p_w3))-((d_perclass/2)*log(2*pi));
end

output_training = [];

for i = 1:N 
    if(desc_w(i,1)>desc_w(i,2))
        if(desc_w(i,1)>desc_w(i,3))
            output_training(i)=1;
        else
            output_training(i)=3;
        end
    else
        if(desc_w(i,2)>desc_w(i,3))
            output_training(i)=2;
        else
            output_training(i)=3;
        end
    end
end

conf_mat_train = zeros(3,3);

for i = 1:N
    conf_mat_train(floor((i-1)/N_perclass)+1,output_training(i)) = conf_mat_train(floor((i-1)/N_perclass)+1,output_training(i))+1;
end

tr_err_w1 = conf_mat_train(1,2)+conf_mat_train(1,3);
tr_err_w2 = conf_mat_train(2,1)+conf_mat_train(2,3);
tr_err_w3 = conf_mat_train(3,1)+conf_mat_train(3,2);

tr_p_err_w1 = tr_err_w1/N_perclass;
tr_p_err_w2 = tr_err_w2/N_perclass;
tr_p_err_w3 = tr_err_w3/N_perclass;

tr_tot_p_err = (tr_err_w1+tr_err_w2+tr_err_w3)/N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Test Data %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

tst_mean = mean(test_set,2);

for i =1:N
    tst_x_mn(:,i) = test_set(:,i) - tst_mean;
end

tst_cov_mat = tst_x_mn*tst_x_mn'./N;

[Tv, Te, Tr] = svd(tst_cov_mat);

for i=1:2
    tst_eig_vct(:,i) = Tv(:,i);
    tst_eig_val(:,i) = Te(:,i);
end

new_test_set = (test_set'*eig_vect)';

for i =1:N
    tst_dat_c1 = new_test_set(:,i) - c_mean_w1;
    tst_dat_c2 = new_test_set(:,i) - c_mean_w2;
    tst_dat_c3 = new_test_set(:,i) - c_mean_w3;
    
    test_disc_c(i,1) = ((-1/2)*(tst_dat_c1)'*(inv(covar_w1))*(tst_dat_c1))-((1/2)*(log(det(covar_w1))))+(log(p_w1))-((d_perclass)*log(2*pi));
    test_disc_c(i,2) = ((-1/2)*(tst_dat_c2)'*(inv(covar_w2))*(tst_dat_c2))-((1/2)*(log(det(covar_w2))))+(log(p_w2))-((d_perclass)*log(2*pi));
    test_disc_c(i,3) = ((-1/2)*(tst_dat_c3)'*(inv(covar_w3))*(tst_dat_c3))-((1/2)*(log(det(covar_w3))))+(log(p_w3))-((d_perclass)*log(2*pi));

end

output_test = [];

for i = 1:N
    if(test_disc_c(i,1)>test_disc_c(i,2))
        if(test_disc_c(i,1)>test_disc_c(i,3))
            output_test(i) = 1;
        else
            output_test(i) = 3;
        end
    else
        if(test_disc_c(i,2)>test_disc_c(i,3))
            output_test(i) = 2;
        else
            output_test(i) = 3;
        end
    end
end

fpt1 = fopen('pcan_training_out.txt','w');
fpt2 = fopen('pcan_test_out.txt','w');
for i = 1:N
    fprintf(fpt1,'%d\n',output_training(i));
    fprintf(fpt2,'%d\n',output_test(i));
end
fclose(fpt1);
fclose(fpt2);

file_id = fopen('pcan_test_out.txt');
output_test1 = textscan(file_id,'%d');
fclose(file_id);

[conf_mat_tst,tst_tot_p_err] = verify_testout(output_test1);

end

