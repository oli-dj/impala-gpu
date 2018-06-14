function [distpath, n_u] = dist_path(SG,tau,varargin)
%DIST_PATH Pseudo-random path that prevents simulation of nodes with no
%   information within the neighbourhood, while favouring distance in order
%   to preferentially get large scale statistics, and not small scale
%   statistics early in the simulation.
%
%   Input:
%       SG       : Simulation grid
%       tau      : data template
%   varargin{1}  : Fraction of path to be calculated as "dist_path"
%   varargin{2}  : Weight on number of informed nodes within template
%   varargin{3}  : Fraction of inverse distance to be considered
%   varargin{4}  : Number of nodes to choose from
%
%   Output:
%       distpath : path as described
%       n_u      : number of uninformed nodes
%
% Note: 2D prototype, 3D can be implemented by using 'convn' instead of
%       'conv2' when performing the convolutions.
%
% Oli D. Johannsson, 2018, oli@johannsson.dk

% Default values
fracDistPath = .1;  % Fraction of path to calculate as dist_path
numFac = .5;        % Weight on number of informed nodes within template
distFac = 1.5;      % Fraction of inverse distance to be considered
n_options = 20;     % Number of nodes to choose from

% Read varargin:
if ~isempty(varargin)
    fracDistPath = varargin{1};
    if length(varargin) > 1
        numFac = varargin{2};
        if length(varargin) > 2
            distFac = varargin{3};
            if length(varargin) > 3
                n_options = varargin{4};
            end
        end
    end
end

[nx, ny] = size(SG);

%Uninformed nodes
uninformed = isnan(SG);

% Number of uninformed nodes
n_u = sum(sum(sum(uninformed)));

% Preallocate path
distpath = zeros(floor(n_u*fracDistPath), 2);

% Template and Inverse Distance Silhuettes
invDistSilhuette = zeros(max(tau)-min(tau) +1);
for i=1:length(tau)
    ind = tau(i,:)+max(tau)+1;
    %templateSilhuette(ind(1),ind(2)) = 1;
    invDistSilhuette(ind(1),ind(2)) = 1./(sqrt(tau(i,1)^2+tau(i,2)^2));
end
% Center
ind = max(tau)+1;
% Make centers of silhuettes large, so visited nodes can't get picked again
invDistSilhuette(ind(1),ind(2)) = 2;

% Allocate array to keep track of informed nodes
informed = zeros(size(SG));
informed(~isnan(SG)) = 1;

% Number of informed nodes
n_i = sum(sum(sum(informed)));

start = 1;
if n_i == 0
        %If no informed nodes exist chose a random starting position
        distpath(1,:) = [randi(nx),randi(ny)];
        informed(distpath(1,1),distpath(1,2)) = 1;
        start = start +1;
end

% The function used to element-wise combine arrays
fun = @(a,b) max(a,b);

%This becomes very expensive for large grids to do every iteration
invDistSumGrid = conv2(informed,invDistSilhuette,'same');

%TODO: Make this work for many informed nodes at start.
invDistGrid = conv2(informed,invDistSilhuette,'same');

for i = start:floor(n_u*fracDistPath)
    
    % Set values back to zeros
    invDistGrid(invDistGrid == n_u) = 0;
    if i ~= start
        %
        lastInformed = zeros(size(SG));
        lastInformed(idx,idy) = 1;
        tempGrid = conv2(lastInformed,invDistSilhuette,'same');
        
        invDistSumGrid = invDistSumGrid + tempGrid;
        invDistGrid = bsxfun(fun,invDistGrid,tempGrid);
        
    end   
   
    invDistGrid(invDistGrid == 0) = n_u; %Set zeros to large number
    
    % Find the InvDistance value of the node furthest away
    furthestInvDistance = min(min(invDistGrid));
    
    % Select all that are less or equal to that distance times a factor
    ids = find(invDistGrid <= distFac*furthestInvDistance);
    invDistances = invDistGrid(ids);
    invSumDistances = invDistSumGrid(ids);
    
    % Sort and 
    %[invDistances, ids_sort] = sort(invDistances);
    %invSumDistances = invSumDistances(ids_sort);
    [invSumDistances, ids_sort] = sort(invSumDistances, 'descend');
    invDistances = invDistances(ids_sort);
    ids = ids(ids_sort);
    
    if length(invDistances) > n_options
        invDistances = invDistances(1:n_options);
        invSumDistances = invSumDistances(1:n_options);
    end
    
    % Caluclate weights, favour small inv-distance AND many counts.
    weights = (1*(1-numFac) + numFac.*invSumDistances)./invDistances;
    %weights = (invSumDistances)./invDistances;
    
    % Normalize
    weights = weights./sum(weights);
    
    % Commulative weights
    weights_cum = cumsum(weights);
    
    % Choose at random with probability decided by weights.
    index = ids(find(weights_cum > rand,1));
    
    [idx,idy] = ind2sub(size(invDistGrid),index);
    informed(idx,idy) = 1;
    distpath(i,:) = [idx,idy];
end

% If dist path calculated for less than all uninformed nodes, add random
% path for the rest.
if fracDistPath < 1
    for i = 1:size(distpath,1)
        SG(distpath(i,1),distpath(i,2)) = 1;
    end
    distpath = vertcat(distpath,rand_path(SG));
end

end