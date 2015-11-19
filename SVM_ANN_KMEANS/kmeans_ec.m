function [cluster_mean, cnt, itr, cluster_new] = kmeans_ec(train_data,c)
m = randi(15000,1,c);
for i=1:c
    cluster_cntrd(:,i) = train_data(:,m(i));
end

% for i=1:c
%         cluster_cntrd(:,i) = sum(train_data(:,(15000/c)*(i-1)+1:(15000/c)*(i))')'./(15000/c);
% end

flag = 1;
iterations = 0;
while ( flag >=0.0001 )
    cluster_count = zeros(1,c);
    % cluster assignment
    clstr = zeros(15000,c);
    for i=1:15000
        for j=1:c
            distance(i,j) = norm(train_data(:,i) - cluster_cntrd(:,j));
        end
        sort_distance = sort(distance(i,:));
        %data index allocation to clusters
        for j=1:c
            if((distance(i,j)-sort_distance(1))==0)
                clstr(i,j)=i;
                cluster_count(j)= cluster_count(j)+1;
            end
        end
    end
    clstr_sum = zeros(4,c);
    %     centroid shifting
    for j=1:c
        for i=1:15000
            if(clstr(i,j)~=0)
                clstr_sum(:,j) = clstr_sum(:,j) + train_data(:,clstr(i,j));
            else
                continue
            end
        end
        cntrd_new(:,j) = clstr_sum(:,j)./cluster_count(j);
    end
    cluster_new = clstr;
    flag = norm(cluster_cntrd - cntrd_new);
    cluster_cntrd = cntrd_new;
    iterations = iterations + 1;
    clear cntrd_new
    clear clstr
end
cluster_mean = cluster_cntrd;
cnt = cluster_count;
itr = iterations;
end
