% MIT License
% 
% Copyright (c) 2018 Andri Ashfahani Mahardhika Pratama
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

clc
clear
close all

%% load file
% datasets are available in this link: https://bit.ly/2lOk0uc
% load weather ; I = 8;
% load electricitypricing; I = 8;
% load sea; I = 3;
% load hyperplane; I = 4;
% load susy; I = 18;
% load Hepmass; I = 28;
% load rlcps; I = 9;
% load permutedmnist; I = 784;
% load kddcup; I = 41;

%% run stacked autonomous deep learning
chunkSize = 500;        % no of data in a batch
epoch = 1;              % no of epoch
alpha_w = 0.0005;       % alpha warning
alpha_d = 0.0001;       % alpha drift
delta   = 0.05;         % pruning layer coefficient delta
[parameter,performance] = ADL(data,I,chunkSize,epoch,alpha_w,alpha_d,...
    delta);
clear data
disp(performance)

% The classification rate in each chunk can be seen in parameter.cr
% The results are the average of all timestamps
