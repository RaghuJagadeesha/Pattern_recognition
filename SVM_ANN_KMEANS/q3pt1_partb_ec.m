clc;
clear all;
close all;

load train_sp2015_v14;
train_data = train_sp2015_v14';


% number of clusters
c1 = 3;
c2 = 4;
c3 = 5;
% randomly initializing the mean

[clst_mn_3, clst_cnt_3, cnt_1, clst_n1] = kmeans_ec(train_data, c1);
[clst_mn_4, clst_cnt_4, cnt_2, clst_n2] = kmeans_ec(train_data, c2);
[clst_mn_5, clst_cnt_5, cnt_3, clst_n3] = kmeans_ec(train_data, c3);

ind_1 = 1;
ind_2 = 1;
ind_3 = 1;

for i=1:15000
    if (clst_n1(i,1) == 0)
        continue
    else
        clust_1(ind_1,:) = train_sp2015_v14(i,:);
        ind_1 = ind_1+1;
    end
end
for i=1:15000
    if (clst_n1(i,2) == 0)
        continue
    else
        clust_2(ind_2,:) = train_sp2015_v14(i,:);
        ind_2 = ind_2+1;
    end
end
for i=1:15000
    if (clst_n1(i,3) == 0)
        continue
    else
        clust_3(ind_3,:) = train_sp2015_v14(i,:);
        ind_3 = ind_3+1;
    end
end
        
ind_1 = 1;
ind_2 = 1;
ind_3 = 1;
ind_4 = 1;

for i=1:15000
    if (clst_n2(i,1) == 0)
        continue
    else
        clust_1_4(ind_1,:) = train_sp2015_v14(i,:);
        ind_1 = ind_1+1;
    end
end
for i=1:15000
    if (clst_n2(i,2) == 0)
        continue
    else
        clust_2_4(ind_2,:) = train_sp2015_v14(i,:);
        ind_2 = ind_2+1;
    end
end
for i=1:15000
    if (clst_n2(i,3) == 0)
        continue
    else
        clust_3_4(ind_3,:) = train_sp2015_v14(i,:);
        ind_3 = ind_3+1;
    end
end

for i=1:15000
    if (clst_n2(i,4) == 0)
        continue
    else
        clust_4_4(ind_4,:) = train_sp2015_v14(i,:);
        ind_4 = ind_4+1;
    end
end

ind_1 = 1;
ind_2 = 1;
ind_3 = 1;
ind_4 = 1;
ind_5 = 1;

for i=1:15000
    if (clst_n3(i,1) == 0)
        continue
    else
        clust_1_5(ind_1,:) = train_sp2015_v14(i,:);
        ind_1 = ind_1+1;
    end
end
for i=1:15000
    if (clst_n3(i,2) == 0)
        continue
    else
        clust_2_5(ind_2,:) = train_sp2015_v14(i,:);
        ind_2 = ind_2+1;
    end
end
for i=1:15000
    if (clst_n3(i,3) == 0)
        continue
    else
        clust_3_5(ind_3,:) = train_sp2015_v14(i,:);
        ind_3 = ind_3+1;
    end
end

for i=1:15000
    if (clst_n3(i,4) == 0)
        continue
    else
        clust_4_5(ind_4,:) = train_sp2015_v14(i,:);
        ind_4 = ind_4+1;
    end
end


for i=1:15000
    if (clst_n3(i,5) == 0)
        continue
    else
        clust_5_5(ind_4,:) = train_sp2015_v14(i,:);
        ind_5 = ind_5+1;
    end
end