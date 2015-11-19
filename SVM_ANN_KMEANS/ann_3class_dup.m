clc
clear all
close all


load train_sp2015_v14
load test_sp2015_v14

test_data = test_sp2015_v14;
train_one=train_sp2015_v14(1:5000,1:4);
train_two=train_sp2015_v14(5001:10000,1:4);
train_three=train_sp2015_v14(10001:15000,1:4);


train=[train_one;train_two;train_three];
[N,d] = size(train_sp2015_v14)
c =3;
lr = 0.75;

for i=1:N/3
    nn_train(3*(i-1)+1,:)=train_sp2015_v14(i,:);
    nn_train(3*(i-1)+2,:)=train_sp2015_v14(i+5000,:);
    nn_train(3*(i-1)+3,:)=train_sp2015_v14(i+10000,:);
end

wt_hdn=linspace(0.021,0.3,36);
wt_ot=linspace(0.025,0.06,27);
bias_ot=0.5;

for i=1:N/3
    target(3*(i-1)+1,:)=[1 0 0];
    target(3*(i-1)+2,:)=[0 1 0];
    target(3*(i-1)+3,:)=[0 0 1];
% outoutput=ones(1,10000);
end


itr=1;
delta_hdn=zeros(15000,9);
while (itr<100)
    for j=1:N
        k=1;
        for i=1:2*d+1
            net_hdn(j,i)=nn_train(j,:)*wt_hdn(k:k+3)'+bias_ot;
            out_hdn(j,i)=1/(1+exp(-net_hdn(j,i)));
            k=k+d;
        end     
        
        k=1;
        for i=1:c    
            net_out(j,i)=out_hdn(j,:)*wt_ot(k:k+8)'+bias_ot;
            out_out(j,i)=1/(1+exp(-net_out(j,i)));
            err(j,i)=(target(j,i)-out_out(j,i));            
            delta_out(j,i)=(target(j,i)-out_out(j,i))*(out_out(j,i)*(1-out_out(j,i)));
            k=k+2*d+1;
        end
        
        k=1;
        for i=1:c
            for p=1:2*d+1                
                delta_wt_ot(j,k)=delta_out(j,i)*out_hdn(j,p)*lr;                
                k=k+1;                
            end            
        end  
        
       
        m=1;      
        for i=1:2*d+1
            for s=1:c
                delta_hdn(j,i)=delta_hdn(j,i)+(out_hdn(j,i)*(1-out_hdn(j,i))*delta_out(j,s)*wt_ot(m));
                m=m+1;
            end
        end
        k=1;
        
        for i=1:2*d+1
            for p=1:d
                delta_wt_hdn(j,k)=delta_hdn(j,i)*nn_train(j,p)*lr;
                k=k+1;              
            end
        end
        
        wt_ot=wt_ot+delta_wt_ot(j,1:(2*d+1)*c);
        wt_hdn=wt_hdn+delta_wt_hdn(j,1:(2*d+1)*d);
        
    end
    TSS=norm(err);
    itr=itr+1;
end



for j=1:N
    k=1;
    for i=1:2*d+1
        net_hdn_fdfwd(j,i)=test_data(j,:)*wt_hdn(k:k+3)'+bias_ot;
        out_hdn_fdfwd(j,i)=1/(1+exp(-net_hdn_fdfwd(j,i)));
        k=k+4;        
    end
	
    k=1;
    for i=1:c
        net_out_fdfwd(j,i)=out_hdn_fdfwd(j,:)*wt_ot(k:k+8)'+bias_ot;
        out_out_fdfwd(j,i)=1/(1+exp(-net_out_fdfwd(j,i)));
        k=k+(2*d+1);
    end
end

for i=1:N
    if (max(out_out_fdfwd(i,:))==out_out_fdfwd(i,1))
        test_output_c(i)=1;
    elseif (max(out_out_fdfwd(i,:))==out_out_fdfwd(i,2))
        test_output_c(i)=2;
    elseif  (max(out_out_fdfwd(i,:))==out_out_fdfwd(i,3))
        test_output_c(i)=3;
    end
end
% *************************************************************************
%                  verifying tested data
% *************************************************************************


ground_truth = [2;3;1;3;1;2];
output_corr = repmat(ground_truth, [2500,1]);


% *****************setting confusion matrix********************************

confMat_result = zeros(3,3);

for i=1:15000
    if (output_corr(i) == 2)
        if test_output_c(i) == 2
            confMat_result(2,2) = confMat_result(2,2) + 1;
        elseif test_output_c(i) == 3
            confMat_result(2,3) = confMat_result(2,3) + 1;
        else
            confMat_result(2,1) = confMat_result(2,1) + 1;
        end
    elseif (output_corr(i) == 3)
        if test_output_c(i) == 2
            confMat_result(3,2) = confMat_result(3,2) + 1;
        elseif test_output_c(i) == 3
            confMat_result(3,3) = confMat_result(3,3) + 1;
        else
            confMat_result(3,1) = confMat_result(3,1) + 1;
        end
    elseif (output_corr(i) == 1)
        if test_output_c(i) == 2
            confMat_result(1,2) = confMat_result(1,2) + 1;
        elseif test_output_c(i) == 3
            confMat_result(1,3) = confMat_result(1,3) + 1;
        else
            confMat_result(1,1) = confMat_result(1,1) + 1;
        end
    end
end

