%% SETUP
TI = [
    1 1 1;
    ];
N = 1;
for i=1:N
    TI = [TI;0 0 1];
end

%Template
tau = [0 -1 0; 0 1 0];

%Generate Impala list
list = populate_impala_list(TI,tau);
print_impala_list(list);

%Vert size
vs = 10000;

%Simulation Grid (i made the grid big, so I only need to call the core 1/vs
% as many times
SG = NaN(vs,3);

%Soft data grid
SDG = NaN(vs,3,2);

%Soft data
SDG(:,1,1) = 0.20;
SDG(:,1,2) = 0.80;

SDG(:,3,1) = 0.3;
SDG(:,3,2) = 0.7;

sillypath = [ [1:vs]',2*ones(vs,1) ];

%Number of soft data to consider
options.num_soft_nc = 2;

options.print = 0;
options.threshold = 0;
options.trimming = 0;
options.trim_size = 0;
options.trim_trigger = 100;
%options.normalize = 0;

cat = unique(TI);


iterations = 1;
%%
SG_tot = zeros(size(SG));
tic
for i = 1:iterations
    %pre-calculate random numbers
    rand_pre = rand(1,vs);
    
    %Run IMPALA core
    [ SG, tauG, stats] = impala_core_gpu_soft(SG, SDG, list, sillypath, tau,...
    rand_pre, cat, options);

    SG_tot = SG_tot + SG;
    fprintf('%i pixels done in %i ms \n',vs,round(1000*toc));
end
fprintf('Total of %i pixels done in %i ms \n',vs*iterations,round(1000*toc));

SG_tot = SG_tot/iterations;
%%
fprintf('IMPALA GPU: P(m_i = 1) = %4.2f \n',mean(SG_tot(:,2)));
