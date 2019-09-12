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

%% main code
function [parameter,performance] = ADL(data,I,chunkSize,epoch,alpha_w,alpha_d,...
    delta)
%% divide the data into nFolds chunks
dataProportion = 1;     % portion of labeled samples, 0-1
fprintf('=========Parallel Autonomous Deep Learning is started=========\n')
[nData,mn] = size(data);
% data_original = data;
M = mn - I;
l = 0;
nFolds       = round(size(data,1)/chunkSize);                 % number of data chunk
chunk_size   = round(nData/nFolds);
round_nFolds = floor(nData/chunk_size);
Data = {};
if round_nFolds == nFolds
    if nFolds   == 1
        Data{1} = data;
    else
        for i=1:nFolds
            l = l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
    end
else
    if nFolds == 1
        Data{1} = data;
    else
        for i=1:nFolds-1
            l=l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
        foldplus = randperm(nFolds-1,1);
        Data{nFolds} = Data{foldplus};
    end
end
buffer_x = [];
buffer_T = [];
tTest = [];
clear data Data1

%% initiate model
K            = 1;          %initial node
parameter.net = netInit([I K M]);

%% initiate node evolving iterative parameters
layer                       = 1;     % number of layer
parameter.ev{1}.layer       = layer;
parameter.ev{1}.kp          = 0;
parameter.ev{1}.miu_x_old   = 0;
parameter.ev{1}.var_x_old   = 0;
parameter.ev{1}.kl          = 0;
parameter.ev{1}.K           = K;
parameter.ev{1}.cr          = 0;
parameter.ev{1}.node        = [];
parameter.ev{1}.BIAS2       = [];
parameter.ev{1}.VAR         = [];
parameter.ev{1}.miu_NS_old  = 0;
parameter.ev{1}.var_NS_old  = 0;
parameter.ev{1}.miu_NHS_old = 0;
parameter.ev{1}.var_NHS_old = 0;
parameter.ev{1}.miumin_NS   = [];
parameter.ev{1}.miumin_NHS  = [];
parameter.ev{1}.stdmin_NS   = [];
parameter.ev{1}.stdmin_NHS  = [];

%% initiate drift detection parameter
alpha   = alpha_d;

%% initiate layer merging iterative parameters
for k3=1:M
    covariance(1,:,k3) = 0;
    covariance(:,1,k3) = 0;
end
covariance_old             = covariance;
threshold                  = delta;      % similarity measure
parameter.prune_list       = 0;
parameter.prune_list_index = [];

count_net = 0;
gap_net = 10000;

%% main loop, prequential evaluation
for t = 1:nFolds
    %% load the data chunk-by-chunk
    x = Data{t}(:,1:I);
    T = Data{t}(:,I+1:mn);
    [bd,~] = size(T);
    clear Data{t}
    
    %% neural network testing
    start_test = tic;
    fprintf('=========Chunk %d of %d=========\n', t, size(Data,2))
    disp('Discriminative Testing: running ...');
    parameter.net.t = t;
    [parameter.net] = testing(parameter.net,x,T,parameter.ev);
    
    %% metrics calculation
    parameter.Loss(t) = parameter.net.loss(parameter.net.index);
    tTest(bd*t+(1-bd):bd*t,:) = parameter.net.sigma;
    acttualLabel(bd*t+(1-bd):bd*t,:) = parameter.net.acttualLabel;
    classPerdiction(bd*t+(1-bd):bd*t,:) = parameter.net.classPerdiction;
    parameter.residual_error(bd*t+(1-bd):bd*t,:) = parameter.net.residual_error;
    parameter.cr(t) = parameter.net.cr;
    
    %% statistical measure
    [performance.ev.f_measure(t,:),performance.ev.g_mean(t,:),performance.ev.recall(t,:),performance.ev.precision(t,:),performance.ev.err(t,:)] = stats(parameter.net.acttualLabel, parameter.net.classPerdiction, M);
    if t == nFolds
        fprintf('=========Parallel Autonomous Deep Learning is finished=========\n')
        break               % last chunk only testing
    end
    parameter.net.test_time(t) = toc(start_test);
    
    %% Layer merging mechanism
    start_train = tic;
    outputcovar = zeros(layer,layer,M);
    for iter = 1:layer
        for iter1 = 1:layer
            if parameter.net.beta(iter) ~= 0 && parameter.net.beta(iter1) ~= 0
                for iter2 = 1:M
                    temporary = cov(parameter.net.as{iter1}(:,iter2),parameter.net.as{iter}(:,iter2));
                    outputcovar(iter,iter1,iter2) = temporary(1,2);
                    covariance (iter,iter1,iter2) = (covariance_old(iter,iter1,iter2)*(t - 1) + (((t - 1)/t)*outputcovar(iter,iter1,iter2)))/t;
                end
            end
        end
    end
    covariance_old = covariance;
    if layer > 1
        merged_list = [];
        for l = 0:layer - 2
            for hh = 1:layer - l - 1
                if parameter.net.beta(end - l) ~= 0 || parameter.net.beta(hh) ~= 0        % only for parallel
                    MCI = zeros(1,M);
                    for o = 1:M
                        pearson = covariance(end - l,hh,o)/sqrt(covariance(end - l,end - l,o)*covariance(hh,hh,o));
                        MCI(o)  = (0.5*(covariance(hh,hh,o) + covariance(end - l,end - l,o)) - sqrt((covariance(hh,hh,o) + covariance(end - l,end - l,o))^(2) - 4*covariance(end - l,end - l,o)*covariance(hh,hh,o)*(1 - pearson^(2))));
                    end
                    if max(abs(MCI)) < threshold
                        if isempty(merged_list)
                            merged_list(1,1) = layer - l;
                            merged_list(1,2) = hh;
                        else
                            No  = find(merged_list(:,1:end - 1) == layer - l, 1);
                            No1 = find(merged_list(:,1:end - 1) == hh, 1);
                            if isempty(No) && isempty(No1)
                                merged_list(end + 1,1) = layer - l;
                                merged_list(end + 1,2) = hh;
                            end
                        end
                        break
                    end
                end
            end
        end
        del_list = [];
        for i = 1:size(merged_list,1)
            noOfHighlyCorrelated = find(merged_list(i,:) == 0, 1);
            if isempty(noOfHighlyCorrelated)
                if parameter.net.beta(merged_list(i,1)) > parameter.net.beta(merged_list(i,2))
                    a = merged_list(i,1);
                    b = merged_list(i,2);
                else
                    b = merged_list(i,1);
                    a = merged_list(i,2);
                end
                del_list = [del_list b];
            end
        end
        if isempty(del_list) == false && parameter.net.beta(del_list) ~= 0
            fprintf('The Hidden Layer no %d is PRUNED around chunk %d\n', del_list, t)
            parameter.net.beta(del_list) = 0;
        end
        parameter.prune_list       = parameter.prune_list + length(del_list);
        parameter.prune_list_index = [parameter.prune_list_index del_list];
    end
    
    %% Drift detection: output space
    if t > 1
        cuttingpoint = 0;
        pp    = length(T);
        F_cut = zeros(pp,1);
        F_cut(parameter.net.wrongClass,:) = 1;
        [Fupper,~] = max(F_cut);
        [Flower,~] = min(F_cut);
        miu_F = mean(F_cut);
        for cut = 1:pp
            miu_G = mean(F_cut(1:cut,:));
            [Gupper,~] = max(F_cut(1:cut,:));
            [Glower,~] = min(F_cut(1:cut,:));
            epsilon_G  = (Gupper - Glower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha)));
            epsilon_F  = (Fupper - Flower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha)));
            if (epsilon_G + miu_G) >= (miu_F + epsilon_F)
                cuttingpoint = cut;
                miu_H = mean(F_cut(cuttingpoint+1:end,:));
                epsilon_D = (Fupper-Flower)*sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)))*log(1/alpha_d));
                epsilon_W = (Fupper-Flower)*sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)))*log(1/alpha_w));
                break
            end
        end
        if cuttingpoint == 0
            miu_H = miu_F;
            epsilon_D = (Fupper - Flower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha_d)));
            epsilon_W = (Fupper - Flower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha_w)));
        end
        if abs(miu_G - miu_H) > epsilon_D && cuttingpoint > 1 && cuttingpoint < pp
            st = 1;
            disp('Drift state: DRIFT');
            layer                      = layer + 1;
            parameter.net.nLayer       = parameter.net.nLayer + 1;
            parameter.net.nHiddenLayer = layer;
            parameter.net.index        = parameter.net.nHiddenLayer;
            fprintf('The new Layer no %d is FORMED around chunk %d\n', layer, t)
            
            %% initiate NN weight parameters
            [ii,~] = size(parameter.net.weight{layer-1});
            parameter.net.weight {layer}  = normrnd(0,sqrt(2/(ii+1)),[1,ii+1]);
            parameter.net.momentum{layer} = zeros(1,ii+1);
            parameter.net.grad{layer}     = zeros(1,ii+1);
            
            %% initiate new classifier weight
            parameter.net.weightSoftmax {layer}  = normrnd(0,1,[M,2]);
            parameter.net.momentumSoftmax{layer} = zeros(M,2);
            parameter.net.gradSoftmax{layer}     = zeros(M,2);
            
            %% initiate new voting weight
            parameter.net.beta(layer)    = 1;
            parameter.net.betaOld(layer) = 1;
            parameter.net.p(layer)       = 1;
            
            %% initiate iterative parameters
            parameter.ev{layer}.layer       = layer;
            parameter.ev{layer}.kl          = 0;
            parameter.ev{layer}.K           = 1;
            parameter.ev{layer}.cr          = 0;
            parameter.ev{layer}.node        = [];
            parameter.ev{layer}.miu_NS_old  = 0;
            parameter.ev{layer}.var_NS_old  = 0;
            parameter.ev{layer}.miu_NHS_old = 0;
            parameter.ev{layer}.var_NHS_old = 0;
            parameter.ev{layer}.miumin_NS   = [];
            parameter.ev{layer}.miumin_NHS  = [];
            parameter.ev{layer}.stdmin_NS   = [];
            parameter.ev{layer}.stdmin_NHS  = [];
            parameter.ev{layer}.BIAS2       = [];
            parameter.ev{layer}.VAR         = [];
            
            %% initiate covariance for rule merging
            for k3=1:M
                covariance(layer,:,k3) = 0;
                covariance(:,layer,k3) = 0;
            end
            covariance_old = covariance;
            
            %% check buffer
            if isempty(buffer_x)
            else
                x = [parameter.net.activity{1}(:,2:end);buffer_x]; % input for discriminative training
                T = [T;buffer_T];
                parameter.net.T = T;
                parameter.net = netFeedForward(parameter.net, x, T);
            end
            buffer_x = [];
            buffer_T = [];
        elseif abs(miu_G - miu_H) >= epsilon_W && abs(miu_G - miu_H) < epsilon_D && st ~= 2
            disp('Drift state: WARNING');
            st       = 2;
            buffer_x = x;
            buffer_T = T;
        else
            st = 3;
            disp('Drift state: STABLE');
            
            %% check buffer
            if isempty(buffer_x)
                
            else
                x  = [buffer_x;x];
                T  = [buffer_T;T];
                parameter.net.T = T;
                parameter.net   = netFeedForward(parameter.net, x, T);
            end
            buffer_x = [];
            buffer_T = [];
        end
    else
        st = 3;
        disp('Drift state: STABLE');
        buffer_x = [];
        buffer_T = [];
    end
    drift(t) = st;
    HL(t) = numel(find(parameter.net.beta ~= 0));
    parameter.wl(t) = parameter.net.index;
    
    %% Discrinimanive training for winning layer
    if st ~= 2
        %         disp('Discriminative Training: running ...');
        parameter = training(parameter,T,epoch,dataProportion);
        %         disp('Discriminative Training: ... finished');
    end
    parameter.net.update_time(t) = toc(start_train);
    
    %% clear current chunk data
    clear Data{t}
    parameter.net.activity = {};
    fprintf('=========Hidden layer number %d was updated=========\n', parameter.net.index)
end

%% statistical measure
[performance.f_measure,performance.g_mean,performance.recall,performance.precision,performance.err] = stats(acttualLabel, classPerdiction, M);

%% save the numerical result
parameter.drift         = drift;
parameter.nFolds        = nFolds;
performance.update_time = [mean(parameter.net.update_time) std(parameter.net.update_time)];
performance.test_time   = [mean(parameter.net.test_time) std(parameter.net.test_time)];
performance.classification_rate = [mean(parameter.cr(2:end)) std(parameter.cr(2:end))];
performance.layer               = [mean(HL) std(HL)];
performance.LayerWeight         = parameter.net.beta;
meanode                         = [];
stdnode                         = [];
for i = 1:parameter.net.nHiddenLayer
    a = nnz(~parameter.net.nodes{i});
    parameter.net.nodes{i} = parameter.net.nodes{i}(a+1:t);
    meanode = [meanode mean(parameter.net.nodes{i})];
    stdnode = [stdnode std(parameter.net.nodes{i})];
end
performance.meanode = meanode;
performance.stdnode = stdnode;
performance.NumberOfParameters = parameter.net.mnop;
parameter.HL = HL;

%% plot the result
subplot(3,1,1)
plot(parameter.cr)
ylim([0 1.1]);
xlim([1 nFolds]);
ylabel('Classification Rate')
subplot(3,1,2)
plot(parameter.Loss)
ylim([0 1.1]);
xlim([1 nFolds]);
ylabel('Loss')
subplot(3,1,3)
plot(HL)
ylabel('No of hidden layer')
xlim([1 nFolds]);
xlabel('chunk');
hold off
end

%% testing phase
function [net] = testing(net, x, T, ev)
%% feedforward
net     = netFeedForward(net, x, T);
[m1,m2] = size(T);
factor  = 0.001;

%% obtain trueclass label
[~,acttualLabel] = max(T,[],2);
net.sigma = zeros(m1,m2);
for t = 1 : m1
    for i = 1 : net.nHiddenLayer
        if net.beta(i) ~= 0
            %% obtain the predicted label
            % note that the layer weight betaOld is fixed
            net.sigma(t,:) = net.sigma(t,:) + net.as{i}(t,:)*net.betaOld(i);
            [~, net.classlabel{i}(t,:)] = max(net.as{i}(t,:),[],2);
            compare = acttualLabel(t,:) - net.classlabel{i}(t,:);
            
            %% train the weighted voting
            if compare ~= 0
                net.p(i) = max(net.p(i)-factor,factor);
                net.beta(i) = max(net.beta(i)*net.p(i),factor);
            elseif compare == 0
                net.p(i) = min(net.p(i)+factor,1);
                net.beta(i) = min(net.beta(i)*(1+net.p(i)),1);
            end
        end
        
        if t == m1
            %% calculate the number of parameter
            if net.beta(i) ~= 0
                [c,d] = size(net.weightSoftmax{i});
                vw = 1;
            else
                c = 0;
                d = 0;
                vw = 0;
            end
            [a,b] = size(net.weight{i});
            nop(i) = a*b + c*d + vw;
            
            %% calculate the number of node in each hidden layer
            net.nodes{i}(net.t) = ev{i}.K;
        end
    end
end
net.nop(net.t) = sum(nop);
net.mnop       = [mean(net.nop) std(net.nop)];

%% update the voting weight
net.beta      = net.beta/sum(net.beta);
net.betaOld   = net.beta;
[~,net.index] = max(net.beta);

%% calculate classification rate
[multiClassProb,classPerdiction] = max(net.sigma,[],2);
net.wrongClass      = find(classPerdiction ~= acttualLabel);
net.cr              = 1 - numel(net.wrongClass)/m1;
net.residual_error  = 1 - multiClassProb;
net.classPerdiction = classPerdiction;
net.acttualLabel    = acttualLabel;
end

%% train the winning layer
function parameter  = training(parameter,y,nEpoch,dataProportion)
[~,bb] = size(parameter.net.weight{parameter.net.index});
grow   = 0;
prune  = 0;

%% initiate performance matrix
ly          = parameter.net.index;
kp          = parameter.ev{1}.kp;
miu_x_old   = parameter.ev{1}.miu_x_old;
var_x_old   = parameter.ev{1}.var_x_old;
kl          = parameter.ev{ly}.kl;
K           = parameter.ev{ly}.K;
node        = parameter.ev{ly}.node;
BIAS2       = parameter.ev{ly}.BIAS2;
VAR         = parameter.ev{ly}.VAR;
miu_NS_old  = parameter.ev{ly}.miu_NS_old;
var_NS_old  = parameter.ev{ly}.var_NS_old;
miu_NHS_old = parameter.ev{ly}.miu_NHS_old;
var_NHS_old = parameter.ev{ly}.var_NHS_old;
miumin_NS   = parameter.ev{ly}.miumin_NS;
miumin_NHS  = parameter.ev{ly}.miumin_NHS;
stdmin_NS   = parameter.ev{ly}.stdmin_NS;
stdmin_NHS  = parameter.ev{ly}.stdmin_NHS;

%% initiate training model
net                    = netInitWinner([1 1 1]);
net.activationFunction = parameter.net.activationFunction;
net.output             = parameter.net.output;

%% substitute the weight to be trained to training model
net.weight{1}   = parameter.net.weight{ly};
net.momentum{1} = parameter.net.momentum{ly};
net.grad{1}     = parameter.net.grad{ly};
net.weight{2}   = parameter.net.weightSoftmax{ly};
net.momentum{2} = parameter.net.momentumSoftmax{ly};
net.grad{2}     = parameter.net.gradSoftmax{ly};

%% load the data for training
x = parameter.net.activity{ly};
[N,I]   = size(x);
s       = RandStream('mt19937ar','Seed',0);
kk      = randperm(s,N);
x       = x(kk,:);
y       = y(kk,:);
nLabeledData = round(dataProportion*N);
x       = x(1:nLabeledData,:);
y       = y(1:nLabeledData,:);
[N,~]   = size(x);

%% xavier initialization
if ly > 1
    n_in = parameter.ev{ly-1}.K;
else
    n_in = parameter.net.initialConfig(1);
end

%% main loop, train the model
for k = 1 : N
    kp = kp + 1;
    kl = kl + 1;
    
    %% Incremental calculation of x_tail mean and variance
    [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,parameter.net.activity{1}(k,:),kp);
    miu_x_old = miu_x;
    var_x_old = var_x;
    
    %% Expectation of z
    py = probit(miu_x,std_x)';
    for ii = 1:parameter.net.index
        if ii == parameter.net.index
            py = sigmf(net.weight{1}*py,[1,0]);
        else
            py = sigmf(parameter.net.weight{ii}*py,[1,0]);
        end
        py = [1;py];
        if ii == 1
            Ey2 = py.^2;
        end
    end
    Ey = py;
    Ez = net.weight{2}*Ey;
    Ez = exp(Ez - max(Ez));
    Ez = Ez./sum(Ez);
    if parameter.net.nHiddenLayer > 1
        py = Ey2;
        for ii = 2:parameter.net.index
            if ii == parameter.net.index
                py = sigmf(net.weight{1}*py,[1,0]);
            else
                py = sigmf(parameter.net.weight{ii}*py,[1,0]);
            end
            py = [1;py];
        end
        Ey2 = py;
    end
    Ez2 = net.weight{2}*Ey2;
    Ez2 = exp(Ez2 - max(Ez2));
    Ez2 = Ez2./sum(Ez2);
    
    %% Network mean calculation
    bias2 = (Ez - y(k,:)').^2;
    ns    = bias2;
    NS    = norm(ns,'fro');
    
    %% Incremental calculation of NS mean and variance
    [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kl);
    miu_NS_old = miu_NS;
    var_NS_old = var_NS;
    miustd_NS  = miu_NS + std_NS;
    miuNS(k,:) = miu_NS;
    if kl <= 1 || grow == 1
        miumin_NS = miu_NS;
        stdmin_NS = std_NS;
    else
        if miu_NS < miumin_NS
            miumin_NS = miu_NS;
        end
        if std_NS < stdmin_NS
            stdmin_NS = std_NS;
        end
    end
    miuminNS(k,:) = miumin_NS;
    miustdmin_NS  = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;
    BIAS2(kp,:)   = miu_NS;
    
    %% growing hidden unit
    if miustd_NS >= miustdmin_NS && kl > 1
        grow            = 1;
        K               = K + 1;
        fprintf('The new node no %d is FORMED around sample %d\n', K, kp)
        node(kp)        = K;
        net.weight{1}   = [net.weight{1};normrnd(0,sqrt(2/(n_in+1)),[1,bb])];
        net.momentum{1} = [net.momentum{1};zeros(1,bb)];
        net.grad{1}     = [net.grad{1};zeros(1,bb)];
        net.weight{2}   = [net.weight{2} normrnd(0,sqrt(2/(K+1)),[parameter.net.initialConfig(end),1])];
        net.momentum{2} = [net.momentum{2} zeros(parameter.net.initialConfig(end),1)];
        net.grad{2}     = [net.grad{2} zeros(parameter.net.initialConfig(end),1)];
        if ly < parameter.net.nHiddenLayer
            [wNext,~]                    = size(parameter.net.weight{ly+1});
            parameter.net.weight{ly+1}   = [parameter.net.weight{ly+1} normrnd(0,sqrt(2/(K+1)),[wNext,1])];
            parameter.net.momentum{ly+1} = [parameter.net.momentum{ly+1} zeros(wNext,1)];
            parameter.net.grad{ly+1}     = [parameter.net.grad{ly+1} zeros(wNext,1)];
        end
    else
        grow     = 0;
        node(kp) = K;
    end
    
    %% Network variance calculation
    var = Ez2 - Ez.^2;
    NHS = norm(var,'fro');
    
    %% Incremental calculation of NHS mean and variance
    [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kl);
    miu_NHS_old = miu_NHS;
    var_NHS_old = var_NHS;
    miustd_NHS  = miu_NHS + std_NHS;
    miuNHS(k,:) = miu_NHS;
    if kl <= I+1 || prune == 1
        miumin_NHS = miu_NHS;
        stdmin_NHS = std_NHS;
    else
        if miu_NHS < miumin_NHS
            miumin_NHS = miu_NHS;
        end
        if std_NHS < stdmin_NHS
            stdmin_NHS = std_NHS;
        end
    end
    miustdmin_NHS  = miumin_NHS + (2.6*exp(-NHS)+1.4)*stdmin_NHS;
    VAR(kp,:)      = miu_NHS;
    
    %% pruning hidden unit
    if grow == 0 && K > 1 && miustd_NHS >= miustdmin_NHS && kl > I + 1
        HS       = Ey(2:end);
        [~,BB]   = min(HS);
        fprintf('The node no %d is PRUNED around sample %d\n', BB, kp)
        prune    = 1;
        K        = K - 1;
        node(kp) = K;
        net.weight{1}(BB,:)   = [];
        net.momentum{1}(BB,:) = [];
        net.grad{1}(BB,:)     = [];
        net.weight{2}(:,BB+1)   = [];
        net.momentum{2}(:,BB+1) = [];
        net.grad{2}(:,BB+1)     = [];
        if ly < parameter.net.nHiddenLayer
            parameter.net.weight{ly+1}(:,BB+1)   = [];
            parameter.net.momentum{ly+1}(:,BB+1) = [];
            parameter.net.grad{ly+1}(:,BB+1)     = [];
        end
    else
        node(kp) = K;
        prune = 0;
    end
    
    %% feedforward
    net = netFeedForwardWinner(net, x(k,:), y(k,:));
    
    %% feedforward #2, executed if there is a hidden node changing
    net = lossBackward(net);
    net = optimizerStep(net);
end

%% iterative learning
if nEpoch > 1
    for iEpoch = 1:nEpoch
        kk = randperm(N);
        x = x(kk,:);
        y = y(kk,:);
        for k = 1 : N
            %% feedforward
            net = netFeedForwardWinner(net, x(k,:), y(k,:));
            
            %% feedforward #2, executed if there is a hidden node changing
            net = lossBackward(net);
            net = optimizerStep(net);
        end
    end
end

%% substitute the weight back to main model
parameter.net.weight{ly}         = net.weight{1};
parameter.net.weightSoftmax{ly}  = net.weight{2};

%% reset momentumCoeff and gradient
parameter.net.momentum{ly}  = net.momentum{1}*0;
parameter.net.grad{ly}      = net.grad{1}*0;
parameter.net.momentumSoftmax{ly} = net.momentum{2}*0;
parameter.net.gradSoftmax{ly}     = net.grad{2}*0;

%% substitute the recursive calculation
parameter.ev{1}.kp           = kp;
parameter.ev{1}.miu_x_old    = miu_x_old;
parameter.ev{1}.var_x_old    = var_x_old;
parameter.ev{ly}.kl          = kl;
parameter.ev{ly}.K           = K;
parameter.ev{ly}.node        = node;
parameter.ev{ly}.BIAS2       = BIAS2;
parameter.ev{ly}.VAR         = VAR;
parameter.ev{ly}.miu_NS_old  = miu_NS_old;
parameter.ev{ly}.var_NS_old  = var_NS_old;
parameter.ev{ly}.miu_NHS_old = miu_NHS_old;
parameter.ev{ly}.var_NHS_old = var_NHS_old;
parameter.ev{ly}.miumin_NS   = miumin_NS;
parameter.ev{ly}.miumin_NHS  = miumin_NHS;
parameter.ev{ly}.stdmin_NS   = stdmin_NS;
parameter.ev{ly}.stdmin_NHS  = stdmin_NHS;
end

%% calculate recursive mean and standard deviation
function [miu,std,var] = meanstditer(miu_old,var_old,x,k)
miu = miu_old + (x - miu_old)./k;
var = var_old + (x - miu_old).*(x - miu);
std = sqrt(var/k);
end

%% calculate probit function
function p = probit(miu,std)
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end

%% stable softmax
function output = stableSoftmax(activation,weight)
output = activation * weight';
output = exp(output - max(output,[],2));
output = output./sum(output, 2);
end

%% initiate neural network
function net = netInit(layer)
net.initialConfig        = layer;
net.nLayer               = numel(net.initialConfig);  %  Number of layer
net.nHiddenLayer         = net.nLayer - 2;   %  number of hidden layer
net.activationFunction   = 'sigmf';          %  Activation functions of hidden layers: 'sigmoid function', 'tanh' and 'relu'.
net.learningRate         = 0.01;             %  learning rate smaller value is preferred
net.momentumCoeff        = 0.95;             %  Momentum coefficient, higher value is preferred
net.outputConnect        = 1;                %  1: connect all hidden layer output to output layer, otherwise: only the last hidden layer is connected to output
net.output               = 'softmax';        %  output layer can be selected as follows: 'sigmoid function', 'softmax function', and 'linear function'

%% initiate weights and weight momentumCoeff for hidden layer
for i = 2 : net.nLayer - 1
    net.weight {i - 1}  = normrnd(0,sqrt(2/(net.initialConfig(i-1)+1)),[net.initialConfig(i),net.initialConfig(i - 1)+1]);
    net.momentum{i - 1} = zeros(size(net.weight{i - 1}));
    net.grad{i - 1}     = zeros(size(net.weight{i - 1}));
    net.c{i - 1}        = normrnd(0,sqrt(2/(net.initialConfig(i-1)+1)),[net.initialConfig(i - 1),1]);
end

%% initiate weights and weight momentumCoeff for output layer
for i = 1 : net.nHiddenLayer
    net.weightSoftmax {i}   = normrnd(0,sqrt(2/(size(net.weight{i},1)+1)),[net.initialConfig(end),net.initialConfig(i+1)+1]);
    net.momentumSoftmax{i}  = zeros(size(net.weightSoftmax{i}));
    net.gradSoftmax{i}      = zeros(size(net.weightSoftmax{i}));
    net.beta(i)             = 1;
    net.betaOld(i)          = 1;
    net.p(i)                = 1;
end
end

function net = netInitWinner(layer)
net.initialConfig   = layer;                         %  winning layer
net.nLayer          = numel(net.initialConfig);      %  Number of layer
net.learningRate                     = 0.01;  %2     %  learning rate, smaller value is preferred
net.momentumCoeff                    = 0.95;         %  Momentum coefficient, higher value is preferred
end

%% feedforward operation
function net = netFeedForward(net, x, output)
nLayer = net.nLayer;
batchSize = size(x,1);
x = [ones(batchSize,1) x];  % by adding 1 to the first coulomn, it means the first coulomn of weight is bias
net.activity{1} = x;        % the first activity is the input itself

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activationFunction
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
    end
    net.activity{iLayer} = [ones(batchSize,1) net.activity{iLayer}];
end

%% propagate to the output layer
for iLayer = 1 : net.nHiddenLayer
    if net.beta(iLayer) ~= 0
        switch net.output
            case 'sigmf'
                net.as{iLayer} = sigmf(net.activity{iLayer + 1} * net.weightSoftmax{iLayer}',[1,0]);
            case 'linear'
                net.as{iLayer} = net.activity{iLayer + 1} * net.weightSoftmax{iLayer}';
            case 'softmax'
                net.as{iLayer} = stableSoftmax(net.activity{iLayer + 1},net.weightSoftmax{iLayer});
        end
        
        %% calculate error
        net.error{iLayer} = output - net.as{iLayer};
        
        %% calculate loss function
        switch net.output
            case {'sigmf', 'linear'}
                net.loss(iLayer) = 1/2 * sum(sum(net.error{iLayer} .^ 2)) / batchSize;
            case 'softmax'
                net.loss(iLayer) = -sum(sum(output .* log(net.as{iLayer}))) / batchSize;
        end
    end
end
end

%% feedforward for a single layer network. It is used to train the winning layer
function net = netFeedForwardWinner(net, input, output)
nLayer = net.nLayer;
m = size(input,1);
net.activity{1} = input;

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activationFunction
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
    end
    net.activity{iLayer} = [ones(m,1) net.activity{iLayer}];  % augment with ones, act as bias multiplier
end

%% propagate to the output layer
switch net.output
    case 'sigmf'
        net.activity{nLayer} = sigmf(net.activity{nLayer - 1} * net.weight{nLayer - 1}',[1,0]);
    case 'linear'
        net.activity{nLayer} = net.activity{nLayer - 1} * net.weight{nLayer - 1}';
    case 'softmax'
        net.activity{nLayer} = stableSoftmax(net.activity{nLayer - 1},net.weight{nLayer - 1});
end

%% calculate error
net.error = output - net.activity{nLayer};
end

%% calculate backpropagation of the network
function net = lossBackward(net)
nLayer = net.nLayer;

%% error backward
switch net.output
    case 'sigmf'
        backPropSignal{nLayer} = - net.error .* (net.activity{nLayer} .* (1 - net.activity{nLayer}));
    case {'softmax','linear'}
        backPropSignal{nLayer} = - net.error;          % dL/dy
end

%% activation backward
for i = (nLayer - 1) : -1 : 2
    switch net.activationFunction
        case 'sigmf'
            actFuncDerivative = net.activity{i} .* (1 - net.activity{i}); % contains b
        case 'tanh'
            actFuncDerivative = 1 - net.activity{i}.^2;
        case 'relu'
            actFuncDerivative = zeros(1,length(net.activity{i}));
            actFuncDerivative(net.activity{i}>0) = 0.1;
    end
    
    if i+1 == nLayer
        backPropSignal{i} = (backPropSignal{i + 1} * net.weight{i}) .* actFuncDerivative;
    else
        backPropSignal{i} = (backPropSignal{i + 1}(:,2:end) * net.weight{i}) .* actFuncDerivative;
    end
end

%% calculate gradient
for i = 1 : (nLayer - 1)
    if i + 1 == nLayer
        net.grad{i} = (backPropSignal{i + 1}' * net.activity{i});
    else
        net.grad{i} = (backPropSignal{i + 1}(:,2:end)' * net.activity{i});
    end
end
end

%% update the weight
function net = optimizerStep(net)
for iLayer = 1 : (net.nLayer - 1)
    grad = net.grad{iLayer};
    grad = net.learningRate * grad;
    net.momentum{iLayer} = net.momentumCoeff*net.momentum{iLayer} + grad;
    grad                 = net.momentum{iLayer};
    
    %% apply the gradient to the weight
    net.weight{iLayer} = net.weight{iLayer} - grad;
end
end

%% calculate performance metrics
function [f_measure,g_mean,recall,precision,err] = stats(f, h, mclass)
%   [f_measure,g_mean,recall,precision,err] = stats(f, h, mclass)
%     @f - vector of true labels
%     @h - vector of predictions on f
%     @mclass - number of classes
%     @f_measure
%     @g_mean
%     @recall
%     @precision
%     @err
%

%     stats.m
%     Copyright (C) 2013 Gregory Ditzler
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

F = index2vector(f, mclass);
H = index2vector(h, mclass);

recall = compute_recall(F, H, mclass);
err = 1 - sum(diag(H'*F))/sum(sum(H'*F));
precision = compute_precision(F, H, mclass);
g_mean = compute_g_mean(recall, mclass);
f_measure = compute_f_measure(F, H, mclass);
end

function g_mean = compute_g_mean(recall, mclass)
g_mean = (prod(recall))^(1/mclass);
end

function f_measure = compute_f_measure(F, H, mclass)
f_measure = zeros(1, mclass);
for c = 1:mclass
    f_measure(c) = 2*F(:, c)'*H(:, c)/(sum(H(:, c)) + sum(F(:, c)));
end
f_measure(isnan(f_measure)) = 1;
end

function precision = compute_precision(F, H, mclass)
precision = zeros(1, mclass);
for c = 1:mclass
    precision(c) = F(:, c)'*H(:, c)/sum(H(:, c));
end
precision(isnan(precision)) = 1;
end

function recall = compute_recall(F, H, mclass)
recall = zeros(1, mclass);
for c = 1:mclass
    recall(c) = F(:, c)'*H(:, c)/sum(F(:, c));
end
recall(isnan(recall)) = 1;
end

function y = index2vector(x, mclass)
y = zeros(numel(x), mclass);
for n = 1:numel(x)
    y(n, x(n)) = 1;
end
end
