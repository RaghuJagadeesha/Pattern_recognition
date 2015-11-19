function [conf_mat_out,tot_p_err] = verify_testout(test_output)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
grnd_truth = [2,3,1,3,1,2];
conf_mat_out = zeros(3,3);
for i = 0:size(test_output{1})-1
    conf_mat_out(grnd_truth(mod(i,6)+1),test_output{1}(i+1)) = conf_mat_out(grnd_truth(mod(i,6)+1),test_output{1}(i+1))+1;          
end
N_perclass = 5000;
err_w1 = conf_mat_out(1,2)+conf_mat_out(1,3);
err_w2 = conf_mat_out(2,1)+conf_mat_out(2,3);
err_w3 = conf_mat_out(3,1)+conf_mat_out(3,2);

p_err_w1 = err_w1/N_perclass;
p_err_w2 = err_w2/N_perclass;
p_err_w3 = err_w3/N_perclass;

tot_p_err = (err_w1+err_w2+err_w3)/(N_perclass*3);

end