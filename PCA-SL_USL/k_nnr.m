function [ conf_mat_1, tst_tot_p_err1, conf_mat_3, tst_tot_p_err3, conf_mat_5, tst_tot_p_err5 ] = k_nnr( train_set, test_set )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[d_t, N_t] = size(train_set);
[d_s, N_s] = size(test_set);
dist_x = [];
for j = 1:N_s
    for i = 1:N_t
        sum = 0;
        for k = 1:d_t
            sum = sum + power(test_set(k,j)-train_set(k,i),2);
        end
        dist_x(i) = sum;
    end
    [sort_dist_x, index] = sort(dist_x);
   
    for idx = [0,1,2]
        count_w1 = 0;
        count_w2 = 0;
        count_w3 = 0;
        for m = 1:(2*idx+1)    
            if(index(m) <= 5000)
                count_w1 = count_w1+1;
            elseif(index(m) >5000 && index(m) <= 10000)
                count_w2 = count_w2+1;
            else
                count_w3 = count_w3+1;
            end
        end
    
        if(count_w1 > count_w2)
            if(count_w1 > count_w3)
                output_test(idx+1,j) = 1;
            else
                output_test(idx+1,j) = 3;
            end
        else
            if(count_w2 > count_w3)
                output_test(idx+1,j) = 2;
            else 
                output_test(idx+1,j) = 3;
            end
        end
    end
end

fpt = fopen('knnr_test_out_1.txt','w');
for i = 1:N_s
    fprintf(fpt,'%d\n',output_test(1,i));
end
fclose(fpt);

file_id = fopen('knnr_test_out_1.txt');
output_test1 = textscan(file_id,'%d');
fclose(file_id);

[conf_mat_1,tst_tot_p_err1] = verify_testout(output_test1);

fpt = fopen('knnr_test_out_3.txt','w');
for i = 1:N_s
    fprintf(fpt,'%d\n',output_test(2,i));
end
fclose(fpt);

file_id = fopen('knnr_test_out_3.txt');
output_test1 = textscan(file_id,'%d');
fclose(file_id);

[conf_mat_3,tst_tot_p_err3] = verify_testout(output_test1);

fpt = fopen('knnr_test_out_5.txt','w');
for i = 1:N_s
    fprintf(fpt,'%d\n',output_test(3,i));
end
fclose(fpt);

file_id = fopen('knnr_test_out_5.txt');
output_test1 = textscan(file_id,'%d');
fclose(file_id);

[conf_mat_5,tst_tot_p_err5] = verify_testout(output_test1);

end

