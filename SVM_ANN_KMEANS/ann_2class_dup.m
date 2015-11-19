clc;
clear all;
close all;

load train_sp2015_v14

fdfwd_train = [train_sp2015_v14(1:2500,:) ;train_sp2015_v14(10001:12500,:)];
fdfwd_tgt = [zeros(2500,1);ones(2500,1)];
d = 4;
N = 10000;
learning_rate = 0.75;
% setting striped inputs

for i=1:N/2
    train(2*(i-1)+1,:) = train_sp2015_v14(i,:);
    train(2*i,:) = train_sp2015_v14(2*(N/2)+i,:);
end


train_data = train;

% d=4 implies that the number of units = 2d+1 = 9
% weights for hidden layer and the output layer

wt_hdn = linspace(0.015,0.12,(2*d+1)*d)';
wt_ot = linspace(0.035,0.08,2*d+1)';
bias_ot = 0.5;     % bias

itr = 1;


% setting target vector
for i=1:N/2
    target(2*(i-1)+1) = 0;
    target(2*i) = 1;
end
% target = [ones(1,5000) zeros(1,5000)];

err = zeros(N,1);
tot_err = 1;


% Neural Net Implementation
while(((itr < 100)))
    for i=1:N
            k=1;
                
        % Hidden layer FF
        for j=1:(2*d+1)
            net_hdn(i,j) = train_data(i,:)*wt_hdn(k:k+3)+bias_ot;
            out_hdn(i,j) = 1/(1+exp(-net_hdn(i,j)));
            k=k+4;
        end
        
        % Output Layer FF

        net_out(i) = out_hdn(i,:)*wt_ot+bias_ot;
        out_out(i) = 1/(1+exp(-net_out(i)));
        err(i) = target(i) - out_out(i);

        % Back propogation
        
        % Output layer
        delta_out(i) = (target(i) - out_out(i))* out_out(i)*(1-out_out(i));
        for j=1:2*d+1
            delta_wt_ot(i,j) = learning_rate*delta_out(i) * out_hdn(i,j);
            delta_hdn(i,j) = out_hdn(i,j)*(1-out_hdn(i,j))*...
                delta_out(i)*wt_ot(j);
        end
        
        % Correction hidden layer
        clear k;
        l=1;
        for j=1:2*d+1
            for k=1:d
            delta_wt_hdn(i,l) = learning_rate*delta_hdn(i,j) * train_data(i,k);
            l=l+1;
            end
        end
        
        % updating weights
        wt_hdn = wt_hdn + delta_wt_hdn(i,:)';
        wt_ot = wt_ot + delta_wt_ot(i,:)';

    end
    itr= itr+1;
    tot_err = norm(err);
    
end

% *************************************************************************
% Feedforward testing of the neural network
% *************************************************************************


for i=1:N/2
        k=1;
        for j=1:2*d+1
            net_hdn_fdfwd(i,j)=fdfwd_train(i,:)*wt_hdn(k:k+3)+bias_ot;
            out_hdn_fdfwd(i,j)=1/(1+exp(-net_hdn_fdfwd(i,j)));
            k=k+4;
        end
        net_out_fdfwd(i)=out_hdn_fdfwd(i,:)*wt_ot+bias_ot;
        out_out_fdfwd(i)=1/(1+exp(-net_out_fdfwd(i)));
end

check=zeros(N/2,1);

for i=1:N/2
    if out_out_fdfwd(i)<=0.5
        check(i)=0;
    elseif out_out_fdfwd(i)>0.5
        check(i)=1;
    end
end
true=0;
false=0;
for i=1:N/2
    if check(i)==fdfwd_tgt(i)
        true=true+1;
    else 
        false=false+1;
    end
end

p_err = false/(N/2)