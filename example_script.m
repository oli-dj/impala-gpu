%% Example script
% Requires mGstat in path, specifically 'mps_template.m' and 'channels.m' 
% https://github.com/cultpenguin/mGstat
rng(1); % Set fixed random seed;

%% LOAD DATA
load('reference3.mat')
soft_data_grid1 = SD_thirty; %Set to 
%soft_data_grid2 = 
soft_data = 1; %use soft data at all
soft_data_combine = 0; %if more than one soft data grid.

%Simulation Grid Size (only set if no softdata is used).
sg_x = 30;
sg_y = 30;

%% Training Image
%[d,h]=read_eas('kasted_data//TI_extended_50x50.eas');
%TI = reshape(d(:,2),632,291)';
TI = channels;

%% GENERAL OPTIONS
num_realizations = 1;
print = 1;
plots = 1;

%% Save options
make_save = 0;
save_setup = 0;
output_folder = 'realizations';
filename_prefix = 'realization';

%% IMPALA OPTIONS

% PATH
% 0: raster, 1:random, 2: random preferential, 3: dist
pathtype = 2; 
I_fac = 4; % Only applies to preferential path

% Data template 
template_length = 512;
template_shape = 1;
options.print = print;

% Min counts to make a cdpf, else use marginal distribution
options.threshold = 5;

% Use GPU instead of CPU for searching the list and calculating cdpf
options.GPU = 1; %Requires CUDA capable GPU

% Max number of non-colocated softdata to use (0, 1, 2, 3 ...)
% Note: Currently only works in GPU implementation
options.num_soft_nc = 5;

% Capping (Max number of conditional data to search for in the SG)
% Set to template_length or more to disable.
options.cap = 20;

% Trimming
options.trimming = 0;
options.trim_size = 5;
options.trim_trigger = 10;
options.min_size = 10;

%% Training Image
dim = length(size(TI));
cat = unique(TI(:))';
num_cat = length(cat);
fprintf("There are %i different facies. \n",num_cat);

%%Load soft data:
if soft_data
    SDG = soft_data_grid1;
    %If soft data grid 1 dimensional 
    if size(SDG,3) == 1
        SDG = 1 - SDG;
        SDG(:,:,2) = 1 - SDG(:,:,1);
    end
end
if soft_data_combine
    %Turn NaNs into flat distribution
    soft_data_grid1(isnan(soft_data_grid1)) = 1/size(SDG,3);
    soft_data_grid2(isnan(soft_data_grid2)) = 1/size(SDG,3);
   
    %Combine probabilities
    SDG = soft_data_grid1.*soft_data_grid2;
    
    %Normalize
    SDG = SDG./sum(SDG,3);
    
    %Remove grid points with no informationentropy
    SDG(SDG == 1/size(SDG,3)) = NaN;
    %TODO: Make work for three or more categories
end

%Simulation grid size
if (sg_x == 0) || (sg_y == 0)
    sg_x = size(SDG,1);
    sg_y = size(SDG,2);
end

%Soft data
if (soft_data > 0)   
    sg_x = size(SDG,1);
    sg_y = size(SDG,2);
    sg_z = size(SDG,3);
else
    switch dim
        case 2
            SDG = NaN(sg_x,sg_y,num_cat);
        case 3
            SDG = NaN(sg_x,sg_y,sg_z,num_cat);
    end
end

switch dim
    case 2
        SG = NaN(sg_x,sg_y);
    case 3
        SG = NaN(sg_x,sg_y,sg_z);
end

if ~exist('SDG','var')
     SDG = NaN([size(SG),length(cat)]);
end

%Template
tau = mps_template(template_length,dim,template_shape);

tic
%Populate pattern library (list)
list = populate_impala_list(TI, tau );
time_elapsed = toc;
if print
    fprintf('Time to populate list: %8.3f seconds.\n', time_elapsed);
    fprintf("List length %i \n",size(list,1));
end

%Display list if small enough
if size(list,1) < 40
    print_impala_list( list );
end

SG_tot = zeros(size(SG));
if plots
    fig_current = figure();
end

%% Simulation
for i = 1:num_realizations
    %Generate new path
    tic
    switch  pathtype
        case 0
            [path, n_u] = raster_path(SG);
        case 1
            [path, n_u] = rand_path(SG);
        case 2
            [path, n_u] = pref_path(SG, SDG, I_fac);
        case 3
            [path, n_u] = dist_path(SG, tau);
    end
    time_elapsed = toc;
    if print
        fprintf('Time to generate random path: %8.3f seconds.\n',...
            time_elapsed);
    end
    
    %Pre-calculate random numbers
    rand_pre = rand(n_u,1);
    
tic;
    if options.GPU
        [SG, tauG, stats] = impala_core_gpu_soft(...
            SG, SDG, list, path, tau, rand_pre, cat, options);
    else
        [SG, tauG, stats] = impala_core(...
            SG, SDG, list, path, tau, rand_pre, cat, options);
    end
    time_elapsed = toc;
    if print
        fprintf('Time to generate realization number %i: %8.3f seconds.\n', i, time_elapsed);
    end
    if make_save
        saveRealization(SG,[filename_prefix num2str(i)],output_folder);
    end
    if plots
        imagesc(gca,SG)
        title(sprintf('Realization number %i',i));
        axis image
        axis ij
        hold on;
        drawnow
    end
    
    SG_tot = SG_tot + SG;
    last_i = i;
    SG = NaN(size(SG)); 
    
    

end
if save_setup
    close all;
    save([output_folder '//' 'setup.mat']);
end
if plots
   SG_image = SG_tot./last_i;
   fig_mean = figure;
   imagesc(SG_image);
   title(sprintf('Realizations 1 through %i',i));
   axis image
   axis ij
   drawnow;
end