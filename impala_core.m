function [ SG, tauG, stats ] = impala_core(SG, SDG, list, path, tau,...
    rand_pre, cat, options)
%IMPALA_CORE Function that runs and IMPALA style MPS-algorithm on CPU
%   Implementation of IMPALA-esque MPS algorithm that searches the list
%   using MATLAB's "find". Supports trimming and colocational soft data
%   only
%
% Inputs:
%  SG:        Simulation grid (2D or 3D)
%  SDG:       Soft Data Grid (SG dim +1)
%  list:      IMPALA list (c and d vectors)
%  path:      precalculated path (random or otherwise)
%  tau:       data template
%  rand_pre:  pre calculated random numbers, must be same length as path.
%  cat:       categories
%  options.threshold:   minimum count in list, else use marginal cpdf.
%         .print:       boolean; 1 shows progress, 0 no print to screen
%         .num_soft_nc  Number nof non-colocated soft data to consider
%                       (default 0, increases processing time dramatically)
%         .trim         boolean, to trim or not
%         .trim_size    how much to trim
%         .trim_trigger how many list misses required to trigger a trim
%
% Outputs
%  SG:        Simlation grid
%  tauG:      local size of data template used
%  stats:     structure with statistics
%
% Oli D. Johannsson, 2017, oli@johannsson.dk

%%
print = options.print;
threshold = options.threshold;
trimming = options.trimming;
trim_size = options.trim_size;

%%
formatspec = 'Time elapsed: %1i seconds. %1i percent done ...\n';
n_u = length(path);
dim = length(size(SG));

%only use dimensions of tau present in simulation grid
tau = tau(:,1:dim);

template_length = length(tau);
tauG = zeros(size(SG));

% Init stats
stats.informed_init = NaN(n_u,1);
stats.informed_final = NaN(n_u,1);
stats.time_elapsed = NaN(n_u,1);

% turn pattern library list into pattern and count matrices
D = cell2mat(list(:,1));
C = cell2mat(list(:,2));

%% Calculate marginal cdpf
marginal_counts = sum(C,1);
% total counts
marginal_counts_tot = sum(marginal_counts);
% probabilities
marginal_probs = marginal_counts./marginal_counts_tot;
% commulative probabilities
marginal_prob_cum = cumsum(marginal_probs);

%% First point along the random path
start = 1;
%If no informed nodes exist
if n_u == length(SG(:))
    result = cat(find(marginal_prob_cum > rand_pre(1),1));
    switch dim
        case 1
            SG(path(start,1)) = result;  
        case 2
            SG(path(start,1),path(start,2)) = result;            
        case 3
            SG(path(start,1),path(start,2),path(start,3)) = result;            
    end
    start = start + 1;
end

%% While uninformed nodes exist (e.g. for rest of random path).

%TODO: Implement 1D support too.
%swith for 2D and 3D:
switch dim
    case 2 %2D
        discards = 0;
        trim = 0;
        for i = start:n_u
            d = NaN(1,template_length);
            % Get data event
            for h = 1:(template_length-trim)
                try
                    d(h) = SG(path(i,1)+tau(h,1),path(i,2)+tau(h,2));
                catch
                    d(h) = NaN;
                end
            end
            
            % find which elements of the data event are informed
            informed = find(~isnan(d));
            stats.informed_init(i) = sum(~isnan(d));
            ind = [];
            % if any informed nodes
            if ~isempty(informed)
                counts_tot = 0;
                while counts_tot < threshold
                    % search list for matches with informed nodes                    
                    ind = find(all(D(:,informed)==d(informed), 2));

                    if isempty(ind)
                        informed = informed(1:end-1);
                    else
                        % sum counts of all facies
                        counts_tot = sum(sum(C(ind,:),1));
                        
                        %if below threshold
                        if counts_tot < threshold
                            ind = [];
                            informed = informed(1:end-1);
                            %TODO: is the below equivalant?
                            % Remove last informed node
                            %d(find(~isnan(d),1,'last')) = NaN;
                        end
                    end
                end
                counts = sum(C(ind,:),1);
                % total counts
                counts_tot = sum(counts);
                %if still below thershold
                if counts_tot < threshold
                    % Draw fropom marginal distribution
                    result = cat(find(marginal_prob_cum > rand_pre(i),1));
                    SG(path(i,1),path(i,2)) = result;
                    
                    % Set data event length to zero
                    tauG(path(i,1),path(i,2)) = 0;
                    
                else
                    % if there is soft data
                    if ~isnan(sum(SDG(path(i,:))))
                        %SD = SDG(path(i,:),:);
                        SD = SDG(path(i,1),path(i,2),:);
                        counts = counts.*SD(:)';
                        counts_tot = sum(counts);
                    end
                    
                    % probabilities
                    probs = counts./counts_tot;
                    
                    % commulative probabilities
                    prob_cum = cumsum(probs);
                    % draw a value and assign
                    SG(path(i,1),path(i,2)) = cat(find(prob_cum >...
                        rand_pre(i),1));
                    
                    % record data event length
                    tauG(path(i,1),path(i,2)) = length(informed);
                    stats.informed_final(i) = sum(~isnan(d));
                    discards = discards + stats.informed_init(i) - ...
                                          stats.informed_final(i);
                end
                
            else
                % Draw from marginal distribution
                SG(path(i,1),path(i,2)) = cat(find(marginal_prob_cum >...
                    rand_pre(i),1));
                
                % Set data event length to zero
                tauG(path(i,1),path(i,2)) = 0;
                stats.informed_final(i) = 0;
            end
            if (print && ~mod(100.*i./n_u,5))
                time_elapsed = toc;
                fprintf(formatspec,round(time_elapsed),round(100*(i/n_u)));
            end
            
            stats.time_elapsed(i) = toc;
            stats.template_length(i) = template_length - trim;
            
            %% Trim function
            if trimming && (discards > 100)
                trim = trim + trim_size;
                discards = 0;
            end
        end
        
    case 3
        for i = start:n_u
            d = zeros(1,template_length);
            % get data event
            for h = 1:template_length
                try
                    %works in 3D
                    d(h) = SG(path(i,1)+tau(h,1),path(i,2)+tau(h,2),...
                        path(i,3)+tau(h,3));
                catch
                    d(h) = NaN;
                end
            end
            
            % find which elements of the data event are informed
            informed = find(~isnan(d));
            ind = [];
            % if any informed nodes
            if ~isempty(informed)
                counts_tot = 0;
                while counts_tot < threshold
                    % search list for matches with informed nodes
                    
                    ind = find(all(D(:,informed)==d(informed), 2));

                    if isempty(ind)
                        informed = informed(1:end-1);
                    else
                        % sum counts of all facies
                        counts_tot = sum(sum(C(ind,:),1));
                        
                        %if below threshold
                        if counts_tot < threshold
                            ind = [];
                            informed = informed(1:end-1);
                        end
                    end
                end
                counts = sum(C(ind,:),1);
                % total counts
                counts_tot = sum(counts);
                %if still below thershold
                if counts_tot < threshold
                    % Draw fropom marginal distribution
                    result = cat(find(marginal_prob_cum > rand_pre(i),1));
                    SG(path(i,1),path(i,2),path(i,3)) = result;
                    
                    % Set data event length to zero
                    tauG(path(i,1),path(i,2),path(i,3)) = 0;
                    
                else
                    
                    % probabilities
                    probs = counts./counts_tot;
                    
                    % if there is soft data
                    if ~isnan(sum(SDG(path(i,:))))
                        probs = probs.*SDG(path(i,:));
                    end
                    
                    % commulative probabilities
                    prob_cum = cumsum(probs);
                    % draw a value and assign
                    SG(path(i,1),path(i,2),path(i,3)) = ...
                        cat(find(prob_cum > rand_pre(i),1));
                    
                    % record data event length
                    tauG(path(i,1),path(i,2),path(i,3)) = length(informed);
                end
                
            else
                % Draw fropom marginal distribution
                SG(path(i,1),path(i,2),path(i,3)) = ...
                    cat(find(marginal_prob_cum > rand_pre(i),1));
                
                % Set data event length to zero
                tauG(path(i,1),path(i,2),path(i,3)) = 0;
            end
            if (print && ~mod(100.*i./n_u,5))
                time_elapsed = toc;
                fprintf(formatspec,round(time_elapsed),round(100*(i/n_u)));
            end
        end
        
end
end

