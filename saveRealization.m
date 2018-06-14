function returnString = saveRealization(SG,varargin)
%SAVEREALIZATION.m
%   Save a simulation grid to file with optional meta data
%
% Inputs:
%   SG: A simulation grid (2D/3D)
%   varargin(1): file name
%   varargin(2): output folder
%   varargin(3): stats structure
%   varargin(4): tbd
%   varargin(5): tbd
% 
%
% Outputs: string stating result of operation.
%

if length(varargin) < 1
    save('savefile.mat','SG');
    returnString = sprintf("Realization saved as '%s'",'savefile.mat');
elseif length(varargin) < 2
    save(varargin{1},'SG');
    returnString = sprintf("Realization saved as '%s'",varargin{1});
elseif length(varargin) < 3
    %Check if directory exists
    if exist(varargin{2},'dir') ~= 7
        %Make directory
        mkdir(varargin{2});
    end
    save([varargin{2} '//' varargin{1}],'SG');
    returnString = sprintf("Realization saved as '%s' in folder '%s'",...
        varargin{1},varargin{2});
end

    


end

