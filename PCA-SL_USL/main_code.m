clc;
clear all;
close all;

load train_sp2015_v14;
train_set = train_sp2015_v14';
load test_sp2015_v14;
test_set = test_sp2015_v14';

file_id = fopen('output.txt');
test_output = textscan(file_id,'%d');
fclose(file_id);

[confusion_mat_tst,p_error] = verify_testout(test_output);
fprintf('Probability of Error : %0.4f \n \n', p_error);

[tr_tot_p_err,confusion_mat_pca_tr,tst_tot_p_err,confusion_mat_pca_ts] = pcan(train_set,test_set);

fprintf('Probability of Error for training set : %0.4f \n \n', tr_tot_p_err);
fprintf('Probability of Error for test set : %0.4f \n \n', tst_tot_p_err);


[confusion_mat_knnr1, error_knnr1,confusion_mat_knnr3, error_knnr3,confusion_mat_knnr5, error_knnr5] = k_nnr(train_set, test_set);
fprintf('Probability of Error for 1-nnr test set : %0.4f \n \n', error_knnr1);
fprintf('Probability of Error for 3-nnr test set : %0.4f \n \n', error_knnr3);
fprintf('Probability of Error for 5-nnr test set : %0.4f \n \n', error_knnr5);

[ conf_matrix_hyper, perror_hyper, conf_mat_tst_hyp, tst_tot_p_err_hyp ] = hyperplanar( train_set, test_set, test_output );
fprintf('Probability of Error for Ho-Kashyap hyperplanar classifier for training set : %0.4f \n \n', perror_hyper);
fprintf('Probability of Error for Ho-Kashyap hyperplanar classifier for test set : %0.4f \n \n', tst_tot_p_err_hyp);
