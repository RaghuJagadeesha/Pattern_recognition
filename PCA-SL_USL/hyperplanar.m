function [ conf_matrix, perror_reduced, conf_mat_tst, tst_tot_p_err ] = hyperplanar( train_set, test_set, test_output )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

train_w1 = train_set(:,1:5000);
train_w2 = train_set(:,5001:10000);
train_w3 = train_set(:,10001:15000);
eta = 0.9;

hc_data1 = [ones(1,5000);train_w1];
hc_data2 = [(-1).*ones(1,10000);  (-1)*[train_w2 train_w3]];


a_12 = ones(5,1);
Y_12 = [hc_data1'; hc_data2'];
count_12 = 0;
b_12 = ones(15000,1);
err_12 = zeros(15000,1);
apr_12 = ones(5,1);
bpr = zeros(15000,1);
stop1 = 0;

while (stop1 == 0)
    bpr = Y_12*apr_12;
    err_12 = bpr - b_12;
    b_12 = b_12 + eta * (err_12 + abs(err_12));
    a_12 = ((Y_12'*Y_12)^-1)*Y_12'*b_12;
    count_12 = count_12 + 1;
    if (Y_12*a_12>0)
        stop1 = 1;
    elseif (apr_12 == a_12)
        stop1 = 2;
    else
        apr_12 = a_12;
        count_12 = count_12 + 1;
    end
end

countneg_1 = 0;
y12 = Y_12 * a_12;
for i=1:10000
    if y12(i)<0
        countneg_1 = countneg_1 + 1;
    end
end

hc_data2 = [ones(1,5000);train_w2];
hc_data3 = [(-1).*ones(1,10000); (-1)*[train_w3 train_w1]];

a_23 = zeros(5,1);
Y_23 = [hc_data2'; hc_data3'];
count_23 = 0;
b_23 = ones(15000,1);
err_23 = zeros(15000,1);
apr_23 = ones(5,1);
stop2 = 0;
while (stop2 == 0)
    bpr = Y_23*apr_23;
    err_23 = bpr - b_23;
    b_23 = b_23 + eta * (err_23 + abs(err_23));
    a_23 = ((Y_23'*Y_23)^-1)*Y_23'*b_23;
    count_23 = count_23 + 1;
        if (Y_23*a_23>0)
        stop2 = 1;
    elseif (apr_23 == a_23)
        stop2 = 2;
    else
        apr_23 = a_23;
        count_23 = count_23 + 1;
    end
end    

y23 = Y_23 * a_23; 

countneg_2 = 0;
y23 = Y_23 * a_23;
for i=1:10000
    if y23(i)<0
        countneg_2 = countneg_2 + 1;
    end
end


hc_data1 = [ones(1,5000);train_w3];
hc_data3 = [(-1).*ones(1,10000);  (-1)*[train_w1 train_w2]];

a_31 = ones(5,1);
Y_31 = [hc_data1'; hc_data3'];
count_31 = 0;
b_31 = ones(15000,1);
err_31 = zeros(15000,1);
apr_31 = ones(5,1);
bpr = zeros(15000,1);
stop3 = 0;


while (stop3 == 0)
    bpr = Y_31*apr_31;
    err_31 = bpr - b_31;
    b_31 = b_31 + eta * (err_31 + abs(err_31));
    a_31 = ((Y_31'*Y_31)^-1)*Y_31'*b_31;
    count_31 = count_31 + 1;
    if (Y_31*a_31>0)
        stop3 = 1;
    elseif (apr_31 == a_31)
        stop3 = 2;
    else
        apr_31 = a_31;
        count_31 = count_31 + 1;
    end
end

y31 = Y_31 * a_31;

countneg_3 = 0;
y31 = Y_31 * a_31;
for i=1:10000
    if y31(i)<0
        countneg_3 = countneg_3 + 1;
    end
end

hyp_out = zeros(15000,3);
hyp_out(:,1) = train_set' * apr_12(2:end,:) ;
hyp_out(:,2) = train_set' * apr_23(2:end,:) ;
hyp_out(:,3) = train_set' * apr_31(2:end,:);

for i=1:15000
    if (hyp_out(i,1) > hyp_out(i,2))
        if (hyp_out(i,1) > hyp_out(i,3))
            output_training(i,1) = 1;
        else
            output_training(i,1) = 3;
        end
    else
        if (hyp_out(i,2) > hyp_out(i,3))
            output_training(i,1) = 2;
        else
            output_training(i,1) = 3;
        end
    end
end

count = 0;
conf_matrix = zeros(3,3);
for i = 1:5000
    if output_training(i,1)~=1
        count = count+1;
        if output_training(i,1) == 2
            conf_matrix(1,2) = conf_matrix(1,2) + 1;
        else 
            if output_training(i,1) == 3
                conf_matrix(1,3) = conf_matrix(1,3) + 1;
            end
        end
    else 
        conf_matrix(1,1) = conf_matrix(1,1)+1;
    end
end
error_class1 = count;

count=0;
for i = 5001:10000
    if output_training(i,1)~=2
        count = count+1;
        if output_training(i,1) == 1
            conf_matrix(2,1) = conf_matrix(2,1) + 1;
        else 
            if output_training(i,1) == 3
                conf_matrix(2,3) = conf_matrix(2,3) + 1;
            end
        end
    else 
        conf_matrix(2,2) = conf_matrix(2,2)+1;
    end
end
error_class2 = count;

count = 0;

for i = 10001:15000
    if output_training(i,1)~=3
        count = count+1;            % counting the error in classification
        if output_training(i,1) == 1
            conf_matrix(3,1) = conf_matrix(3,1) + 1;
        else 
            if output_training(i,1) == 2
                conf_matrix(3,2) = conf_matrix(3,2) + 1;
            end
        end
    else 
        conf_matrix(3,3) = conf_matrix(3,3)+1;
    end
end
error_class3 = count;

perror_reduced = (error_class1 + error_class2 + error_class3)/15000;

hyp_out_test = zeros(15000,3);
hyp_out_test(:,1) = test_set' * apr_12(2:end,:);
hyp_out_test(:,2) = test_set' * apr_23(2:end,:);
hyp_out_test(:,3) = test_set' * apr_31(2:end,:);


for i=1:15000
    if (hyp_out_test(i,1) > hyp_out_test(i,2))
        if (hyp_out_test(i,1) > hyp_out_test(i,3))
            output_test(i,1) = 1;
        else
            output_test(i,1) = 3;
        end
    else
        if (hyp_out_test(i,2) > hyp_out_test(i,3))
            output_test(i,1) = 2;
        else
            output_test(i,1) = 3;
        end
    end
end

fpt = fopen('hyperplanar_test_out.txt','w');
for i = 1:15000
    fprintf(fpt,'%d\n',output_test(i,1));
end
fclose(fpt);

file_id = fopen('hyperplanar_test_out.txt');
output_test1 = textscan(file_id,'%d');
fclose(file_id);

[conf_mat_tst,tst_tot_p_err] = verify_testout(output_test1);

end