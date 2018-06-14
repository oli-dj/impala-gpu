fig1 = figure;

%TI = channels;

SG1 = SGnotrim;
SG2 = SGtrim;

labelSG1 = 'Regular';
labelSG2 = 'Trimming'; 

cat = unique(SG);
num_cat = length(cat);

% 1 Point Statistics
[N_TI,~] = histcounts(TI,'BinMethod','integers',...
    'Normalization','probability');
[N_SG1,~] = histcounts(SG1,'BinMethod','integers',...
    'Normalization','probability');
[N_SG2,EDGES] = histcounts(SG2,'BinMethod','integers',...
    'Normalization','probability');

% 2 Point statistics
tau = mps_template(2);
list_2_point_SG1 = populate_impala_list(SG1,tau);
N_2_SG1 = sum(cell2mat(list_2_point_SG1(:,2)),2)./length(SG1(:));

list_2_point_SG2 = populate_impala_list(SG2,tau);
N_2_SG2 = sum(cell2mat(list_2_point_SG2(:,2)),2)./length(SG2(:));

list_2_point_TI = populate_impala_list(TI,tau);
N_2_TI = sum(cell2mat(list_2_point_TI(:,2)),2)./length(TI(:));

% N Point statistics
N = 8;
tau = mps_template(N);
impala_list_N_point_TI = populate_impala_list(TI,tau);
impala_list_N_point_SG1 = populate_impala_list(SG1,tau);
impala_list_N_point_SG2 = populate_impala_list(SG2,tau);

list_N_point_TI = cell2mat(impala_list_N_point_TI(:,1));
list_N_point_SG1 = cell2mat(impala_list_N_point_SG1(:,1));
list_N_point_SG2 = cell2mat(impala_list_N_point_SG2(:,1));

list_union=union(list_N_point_SG1,list_N_point_TI,'rows');
list_union=union(list_N_point_SG2,list_union,'rows');

counts_N_point_TI = cell2mat(impala_list_N_point_TI(:,2));
counts_N_point_SG1 = cell2mat(impala_list_N_point_SG1(:,2));
counts_N_point_SG2 = cell2mat(impala_list_N_point_SG2(:,2));

counts = zeros(length(list_union),3);

for i = 1:length(list_union)
    [LIA,LOB] = ismember(list_union(i,:),list_N_point_TI,'rows');    
    if LIA
        counts(i,1) = sum(counts_N_point_TI(LOB,:));
    end
    [LIA,LOB] = ismember(list_union(i,:),list_N_point_SG1,'rows');
    if LIA
        counts(i,2) = sum(counts_N_point_SG1(LOB,:));
    end
    [LIA,LOB] = ismember(list_union(i,:),list_N_point_SG2,'rows');
    if LIA
        counts(i,3) = sum(counts_N_point_SG2(LOB,:));
    end
end

% Normalize
counts(:,1) =  counts(:,1)./sum(counts(:,1),1);
counts(:,2) =  counts(:,2)./sum(counts(:,2),1);
counts(:,3) =  counts(:,3)./sum(counts(:,3),1);

subplot(2,2,1)
hb = bar(100.*[N_TI',N_SG1',N_SG2']);
title('One-point statistics');
ylabel('Pattern Probability [\%]');
xlabel('Pattern number');
%legend('Training  Image',labelSG1,labelSG2);

%hb(1).FaceColor = [3,131,107]./255;
hb(1).FaceColor = [220,190,50]./255;
hb(2).FaceColor = [0,89,123]./255;
hb(3).FaceColor = [46,159,129]./255;


subplot(2,2,2)
hb2 = bar(100.*[N_2_TI,N_2_SG1,N_2_SG2]);
title('Two-point statistics');
%ylabel('Pattern Probability [\%]');
xlabel('Pattern number');
legend({'Training  Image',labelSG1,labelSG2});
legend('boxoff');
hb2(1).FaceColor = [220,190,50]./255;
hb2(2).FaceColor = [0,89,123]./255;
hb2(3).FaceColor = [46,159,129]./255;

subplot(2,2,[3 4]);
hb3 = bar(100.*counts);
set(gca, 'yscale','log');
title('Six-point statistics');
ylabel('Pattern Probability [\%]');
xlabel('Pattern number');
hb3(1).FaceColor = [220,190,50]./255;
hb3(2).FaceColor = [0,89,123]./255;
hb3(3).FaceColor = [46,159,129]./255;
%legend('Training  Image',labelSG1,labelSG2);

