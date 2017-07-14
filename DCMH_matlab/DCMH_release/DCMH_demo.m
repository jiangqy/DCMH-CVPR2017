function DCMH_demo()
addpath(fullfile('./utils/'));
dataname = 'MIRFLICKR-25K';

% %% preprocessing: download data and pretrained VGG-F net.
% preprocessing;

% load data
load ./data/FLICKR-25K.mat YAll images LAll;
numData = size(LAll, 1);

% split data 
param = gen_index_MIRFLICKR_25K(numData);        

%% parameter setting
param.logfile = ['./result/DCMH_' dataname '_' datestr(now) '.mat'];
param.method = 'DCMH';

param.maxIter = 500;
param.lr_txt = logspace(-1.5,-3,param.maxIter);
param.lr_img = logspace(-1.5,-3,param.maxIter);
param.gamma = 1;
param.eta = 1;
param.batch_size = 128;

dataset.L01 = LAll;
dataset.Y = normZeroMean(YAll);
dataset.X = images;

bits = [16, 32, 64];
nb = numel(bits);

%% training and evaluating DCMH
evaluation = cell(1, nb);
for i = 1: nb
    param.bit = bits(i);
    fprintf('...................................\n');
    fprintf('...dataset: %s\n', dataname);
        
    fprintf('...method: %s\n', param.method);
    fprintf('...bit: %d\n', param.bit);
    result = process_DCMH(dataset, param);
    evaluation{i} = result;
    evaluation{i}.bit = param.bit;
end
save(param.logfile, 'evaluation', 'param');
end

function param = gen_index_MIRFLICKR_25K(numData)
R = randperm(numData);
numQuery = 2000;

param.indexQuery = R(1: numQuery);
R(1: numQuery) = [];
param.indexRetrieval = R;
param.indexTrain = R(1: 10000);
end
