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

function [parameter,performance] = ADL(data,I)
%% divide the data into nFolds chunks
fprintf('=========Parallel Autonomous Deep Learning is started=========\n')
[nData,mn] = size(data);
M = mn - I;
l = 0;
nFolds       = round(length(data)/500);                 % number of data chunk
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
parameter.nn = netconfig([I K M]);

%% initiate node evolving iterative parameters
layer                       = 1;     % number of layer
parameter.ev{1}.layer       = layer;
parameter.ev{1}.kp          = 0;
parameter.ev{1}.miu_x_old   = 0;
parameter.ev{1}.var_x_old   = 0;
parameter.ev{1}.kl          = 0;
parameter.ev{1}.K           = K;
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
alpha_w = 0.0005;
alpha_d = 0.0001;
alpha   = 0.0001;

%% initiate layer merging iterative parameters
for k3=1:M
    covariance(1,:,k3) = 0;
    covariance(:,1,k3) = 0;
end
covariance_old             = covariance;
threshold                  = 0.05;
parameter.prune_list       = 0;
parameter.prune_list_index = [];

%% main loop, prequential evaluation
for t = 1:nFolds
    %% load the data chunk-by-chunk
    x = Data{t}(:,1:I);
    T = Data{t}(:,I+1:mn);
    [bd,~] = size(T);
    tTarget(bd*t+(1-bd):bd*t,:) = T;
    clear Data{t}
    
    %% neural network testing
    start_test = tic;
    fprintf('=========Chunk %d of %d=========\n', t, size(Data,2))
    disp('Discriminative Testing: running ...');
    parameter.nn.t = t;
    parameter.nn = nettestparallel(parameter.nn,x,T,parameter.ev);
    
    %% metrics calculation
    parameter.Loss(t) = parameter.nn.L(parameter.nn.index);
    tTest(bd*t+(1-bd):bd*t,:) = parameter.nn.sigma;
    act(bd*t+(1-bd):bd*t,:) = parameter.nn.act;
    out(bd*t+(1-bd):bd*t,:) = parameter.nn.out;
    parameter.cr(t) = parameter.nn.cr;
    ClassificationRate(t) = mean(parameter.cr);
    fprintf('Classification rate %d\n', ClassificationRate(t))
    disp('Discriminative Testing: ... finished');
    
    %% statistical measure
    [performance.ev.f_measure(t,:),performance.ev.g_mean(t,:),performance.ev.recall(t,:),performance.ev.precision(t,:),performance.ev.err(t,:)] = stats(parameter.nn.act, parameter.nn.out, M);
    if t == nFolds - 1
        fprintf('=========Parallel Autonomous Deep Learning is finished=========\n')
        break               % last chunk only testing
    end
    parameter.nn.test_time(t) = toc(start_test);
    
    %% Layer merging mechanism
    start_train = tic;
    outputcovar = zeros(layer,layer,M);
    for iter = 1:layer
        for iter1 = 1:layer
            if parameter.nn.beta(iter) ~= 0 && parameter.nn.beta(iter1) ~= 0
                for iter2 = 1:M
                    temporary = cov(parameter.nn.as{iter1}(:,iter2),parameter.nn.as{iter}(:,iter2));
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
                if parameter.nn.beta(end - l) ~= 0 || parameter.nn.beta(hh) ~= 0        % only for parallel
                    MCI = zeros(1,M);
                    for o = 1:M
                        pearson = covariance(end - l,hh,o)/sqrt(covariance(end - l,end - l,o)*covariance(hh,hh,o));
                        MCI(o)  = (0.5*(covariance(hh,hh,o) +   covariance(end - l,end - l,o)) - sqrt((covariance(hh,hh,o) + covariance(end - l,end - l,o))^(2) - 4*covariance(end - l,end - l,o)*covariance(hh,hh,o)*(1 - pearson^(2))));
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
            No2 = find(merged_list(i,:) == 0, 1);
            if isempty(No2)
                if parameter.nn.beta(merged_list(i,1)) > parameter.nn.beta(merged_list(i,2))
                    a = merged_list(i,1);
                    b = merged_list(i,2);
                else
                    b = merged_list(i,1);
                    a = merged_list(i,2);
                end
                del_list = [del_list b];
            end
        end
        if isempty(del_list) == false && parameter.nn.beta(del_list) ~= 0
            fprintf('The Hidden Layer no %d is PRUNED around chunk %d\n', del_list, t)
            parameter.nn.beta(del_list) = 0;
        end
        parameter.prune_list = parameter.prune_list + length(del_list);
        parameter.prune_list_index = [parameter.prune_list_index del_list];
    end
    
    %% Drift detection: output space
    if t > 1
        cuttingpoint = 0;
        pp = length(T);
        F_cut = zeros(pp,1);
        F_cut(parameter.nn.bad,:) = 1;
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
            layer = layer + 1;
            parameter.nn.n  = parameter.nn.n + 1;
            parameter.nn.hl = layer;
            fprintf('The new Layer no %d is FORMED around chunk %d\n', layer, t)
            
            %% initiate NN weight parameters
            [ii,~] = size(parameter.nn.W{layer-1});
            parameter.nn.W {layer} = (rand(1,ii+1) - 0.5) * 2 * 4 * sqrt(6 / (M + ii));
            parameter.nn.vW{layer} = zeros(1,ii+1);
            parameter.nn.dW{layer} = zeros(1,ii+1);
            
            %% initiate new classifier weight
            parameter.nn.Ws {layer} = (rand(M,2) - 0.5) * 2 * 4 * sqrt(6 / (M + 2));
            parameter.nn.vWs{layer} = zeros(M,2);
            parameter.nn.dWs{layer} = zeros(M,2);
            
            %% initiate new voting weight
            parameter.nn.beta(layer)    = 1;
            parameter.nn.betaOld(layer) = 1;
            parameter.nn.p(layer)       = 1;
            
            %% initiate iterative parameters
            parameter.ev{layer}.layer       = layer;
            parameter.ev{layer}.kl          = 0;
            parameter.ev{layer}.K           = 1;
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
                h = parameter.nn.a{end}(:,2:end);
                z = T;
            else
                buffer_x = netffhl(parameter.nn,buffer_x);
                h = [buffer_x(:,2:end);parameter.nn.a{end}(:,2:end)];
                z = [buffer_T;T];
            end
            
            %% discriminative training for new layer
            disp('Discriminative Training for new layer: running ...');
            parameter = nettrainsingle(parameter,h,z);
            disp('Discriminative Training for new layer: ... finished');
            buffer_x = [];
            buffer_T = [];
        elseif abs(miu_G - miu_H) >= epsilon_W && abs(miu_G - miu_H) < epsilon_D
            st = 2;
            disp('Drift state: WARNING');
            buffer_x = x;
            buffer_T = T;
        else
            st = 3;
            disp('Drift state: STABLE');
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
    HL(t) = numel(find(parameter.nn.beta ~= 0));
    parameter.wl(t) = parameter.nn.index;
    
    %% Discrinimanive training for winning layer
    if st ~= 1
        disp('Discriminative Training: running ...');
        parameter = nettrainparallel(parameter,T);
        disp('Discriminative Training: ... finished');
    end
    parameter.nn.update_time(t) = toc(start_train);
    
    %% clear current chunk data
    clear Data{t}
    parameter.nn.a = {};
    fprintf('=========Hidden layer number %d was updated=========\n', parameter.nn.index)
end
clc

%% statistical measure
[performance.f_measure,performance.g_mean,performance.recall,performance.precision,performance.err] = stats(act, out, M);

%% save the numerical result
parameter.drift = drift;
parameter.nFolds = nFolds;
performance.update_time = [mean(parameter.nn.update_time) std(parameter.nn.update_time)];
performance.test_time = [mean(parameter.nn.test_time) std(parameter.nn.test_time)];
performance.classification_rate = [mean(parameter.cr(2:end)) std(parameter.cr(2:end))];
performance.layer = [mean(HL) std(HL)];
performance.LayerWeight = parameter.nn.beta;
meanode = [];
stdnode = [];
for i = 1:parameter.nn.hl
    a = nnz(~parameter.nn.nodes{i});
    parameter.nn.nodes{i} = parameter.nn.nodes{i}(a+1:t);
    meanode = [meanode mean(parameter.nn.nodes{i})];
    stdnode = [stdnode std(parameter.nn.nodes{i})];
end
performance.meanode = meanode;
performance.stdnode = stdnode;
performance.NumberOfParameters = parameter.nn.mnop;
parameter.HL = HL;

%% plot the result
subplot(3,1,1)
plot(ClassificationRate)
ylim([0 1.1]);
xlim([1 nFolds]);
ylabel('Classification Rate')
subplot(3,1,2)
plot(parameter.cr)
ylim([0 1.1]);
xlim([1 nFolds]);
ylabel('Classification Rate (t)')
subplot(3,1,3)
plot(HL)
ylabel('No of Layer')
xlim([1 nFolds]);
xlabel('chunk');
hold off
% figure
% plotconfusion(tTarget(2:end,:)',tTest(2:end,:)');

%% display the results

end

function nn = nettestparallel(nn, x, T, ev)
%% feedforward
nn = netfeedforward(nn, x, T);
[m1,m2] = size(T);

%% obtain trueclass label
[~,act] = max(T,[],2);

%% obtain the class label
nn.sigma = zeros(m1,m2);
for t = 1 : m1
    for i = 1 : nn.hl
        if nn.beta(i) ~= 0
            nn.sigma(t,:) = nn.sigma(t,:) + nn.as{i}(t,:)*nn.betaOld(i);
            [~, nn.classlabel{i}(t,:)] = max(nn.as{i}(t,:),[],2);
            compare = act(t,:) - nn.classlabel{i}(t,:);
            
            %% train the weighted voting
            if compare ~= 0
                nn.beta(i) = max(nn.beta(i)*nn.p(i),0);
                nn.p(i) = max(nn.p(i)-0.01,0);
            elseif compare == 0
                nn.beta(i) = min(nn.beta(i)*(1+nn.p(i)),1);
                nn.p(i) = min(nn.p(i)+0.01,1);
            end
        end
        
        if t == m1
            %% calculate the number of parameter
            if nn.beta(i) ~= 0
                [c,d] = size(nn.Ws{i});
                vw = 1;
            else
                c = 0;
                d = 0;
                vw = 0;
            end
            [a,b] = size(nn.W{i});
            nop(i) = a*b + c*d + vw;
            
            %% calculate the number of node in each hidden layer
            nn.nodes{i}(nn.t) = ev{i}.K;
        end
    end
    nn.beta = nn.beta/sum(nn.beta);
end
nn.nop(nn.t) = sum(nop);
nn.mnop = [mean(nn.nop) std(nn.nop)];

%% update the voting weight
nn.betaOld = nn.beta;
[~,nn.index] = max(nn.beta);

%% calculate classification rate
[~,out] = max(nn.sigma,[],2);
nn.bad = find(out ~= act);
nn.cr = 1 - numel(nn.bad)/m1;
nn.out = out;
nn.act = act;
end

function nn = netfeedforward(nn, x, y)
n = nn.n;
m = size(x,1);
x = [ones(m,1) x];      % by adding 1 to the first coulomn, it means the first coulomn of W is bias
nn.a{1} = x;            % the first activity is the input itself

%% feedforward from input layer through all the hidden layer
for i = 2 : n-1
    switch nn.activation_function
        case 'sigm'
            nn.a{i} = sigmf(nn.a{i - 1} * nn.W{i - 1}',[1,0]);
        case 'relu'
            nn.a{i} = max(nn.a{i - 1} * nn.W{i - 1}',0);
    end
    nn.a{i} = [ones(m,1) nn.a{i}];
end

%% propagate to the output layer
for i = 1 : nn.hl
    if nn.beta(i) ~= 0
        switch nn.output
            case 'sigm'
                nn.as{i} = sigmf(nn.a{i + 1} * nn.Ws{i}',[1,0]);
            case 'linear'
                nn.as{i} = nn.a{i + 1} * nn.Ws{i}';
            case 'softmax'
                nn.as{i} = nn.a{i + 1} * nn.Ws{i}';
                nn.as{i} = exp( nn.as{i} - max(nn.as{i},[],2));
                nn.as{i} = nn.as{i}./sum(nn.as{i}, 2);
        end
        
        %% calculate error
        nn.e{i} = y - nn.as{i};
        
        %% calculate loss function
        switch nn.output
            case {'sigm', 'linear'}
                nn.L(i) = 1/2 * sum(sum(nn.e .^ 2)) / m;
            case 'softmax'
                nn.L(i) = -sum(sum(y .* log(nn.as{i}))) / m;
        end
    end
end
end

function parameter  = nettrainsingle(parameter,x,y)
[~,bb] = size(parameter.nn.W{parameter.nn.hl});
grow = 0;
prune = 0;

%% initiate performance matrix
ly          = parameter.nn.hl;
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
net = netconfigtrain([1 1 1]);

%% substitute the weight to be trained to training model
net.activation_function = parameter.nn.activation_function;
net.W{1}  = parameter.nn.W{ly};
net.vW{1} = parameter.nn.vW{ly};
net.dW{1} = parameter.nn.dW{ly};
net.W{2}  = parameter.nn.Ws{ly};
net.vW{2} = parameter.nn.vWs{ly};
net.dW{2} = parameter.nn.dWs{ly};

%% load data
[N,I] = size(x);
kk    = randperm(N);
x     = [ones(N,1) x(kk,:)];
y     = y(kk,:);


%% main loop, train the model
for k = 1 : N
    kp = kp + 1;
    kl = kl + 1;
    
    %% feedforward #1
    net = netffsingle(net, x(k,:), y(k,:));
    
    %% Incremental calculation of x_tail mean and variance
    if k <= size(parameter.nn.a{1},1)
        [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,parameter.nn.a{1}(k,:),kp);
        miu_x_old = miu_x;
        var_x_old = var_x;
        
        %% Expectation of z
        py = probit(miu_x,std_x)';
        for ii = 1:parameter.nn.hl
            if ii == parameter.nn.hl
                py = sigmf(net.W{1}*py,[1,0]);
            else
                py = sigmf(parameter.nn.W{ii}*py,[1,0]);
            end
            py = [1;py];
            if ii == 1
                Ey2 = py.^2;
            end
        end
        Ey = py;
        Ez = net.W{2}*Ey;
        Ez = exp(Ez);
        Ez = Ez./sum(Ez);
        if parameter.nn.hl > 1
            py = Ey2;
            for ii = 2:parameter.nn.hl
                if ii == parameter.nn.hl
                    py = sigmf(net.W{1}*py,[1,0]);
                else
                    py = sigmf(parameter.nn.W{ii}*py,[1,0]);
                end
                py = [1;py];
            end
            Ey2 = py;
        end
        Ez2 = net.W{2}*Ey2;
        Ez2 = exp(Ez2);
        Ez2 = Ez2./sum(Ez2);
        
        %% Network mean calculation
        bias2 = (Ez - y(k,:)').^2;
        ns = bias2;
        NS = norm(ns,'fro');
        
        %% Incremental calculation of NS mean and variance
        [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kp);
        miu_NS_old = miu_NS;
        var_NS_old = var_NS;
        miustd_NS = miu_NS + std_NS;
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
        miustdmin_NS = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;
        BIAS2(kp,:) = miu_NS;
        
        %% growing hidden unit
        if miustd_NS >= miustdmin_NS && kl > 1
            grow = 1;
            K = K + 1;
            fprintf('The new node no %d is FORMED around sample %d\n', K, k)
            node(kp)  = K;
            net.W{1}  = [net.W{1};(rand(1,bb) - 0.5)*2*4*sqrt(6/(1 + bb))];
            net.vW{1} = [net.vW{1};zeros(1,bb)];
            net.dW{1} = [net.dW{1};zeros(1,bb)];
            net.W{2}  = [net.W{2} rand(parameter.nn.size(end),1)];
            net.vW{2} = [net.vW{2} zeros(parameter.nn.size(end),1)];
            net.dW{2} = [net.dW{2} zeros(parameter.nn.size(end),1)];
        else
            grow = 0;
            node(kp) = K;
        end
        
        %% Network variance calculation
        var = Ez2 - Ez.^2;
        NHS = norm(var,'fro');
        
        %% Incremental calculation of NHS mean and variance
        [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kp);
        miu_NHS_old = miu_NHS;
        var_NHS_old = var_NHS;
        miustd_NHS = miu_NHS + std_NHS;
        miuNHS(k,:) = miu_NHS;
        if kl <= I + 1 || prune == 1
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
        miustdmin_NHS = miumin_NHS + (2.6*exp(-NHS)+1.4)*stdmin_NHS;
        VAR(kp,:) = miu_NHS;
        
        %% pruning hidden unit
        if grow == 0 && K > 1 && miustd_NHS >= miustdmin_NHS && kl > I + 1
            HS = Ey(2:end);
            [~,BB] = min(HS);
            fprintf('The node no %d is PRUNED around sample %d\n', BB, k)
            prune = 1;
            K = K - 1;
            node(kp) = K;
            net.W{1}(BB,:)  = [];
            net.vW{1}(BB,:) = [];
            net.dW{1}(BB,:) = [];
            net.W{2}(:,BB+1)  = [];
            net.vW{2}(:,BB+1) = [];
            net.dW{2}(:,BB+1) = [];
        else
            node(kp) = K;
            prune = 0;
        end
        
        if grow == 1 || prune == 1
            net = netffsingle(net, x(k,:), y(k,:));
        end
    end
    
    %% feedforward #2, executed if there is a hidden node changing
    net = netbackpropagation(net);
    net = netupdate(net);
end

%% substitute the weight back to main model
parameter.nn.W{ly}   = net.W{1};
parameter.nn.vW{ly}  = net.vW{1};
parameter.nn.dW{ly}  = net.dW{1};
parameter.nn.Ws{ly}  = net.W{2};
parameter.nn.vWs{ly} = net.vW{2};
parameter.nn.dWs{ly} = net.dW{2};

%% substitute the recursive calculation
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

function parameter  = nettrainparallel(parameter,y)
[~,bb] = size(parameter.nn.W{parameter.nn.index});
grow = 0;
prune = 0;

%% initiate performance matrix
ly          = parameter.nn.index;
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
net = netconfigtrain([1 1 1]);
net.activation_function = parameter.nn.activation_function;

%% substitute the weight to be trained to training model
net.W{1}  = parameter.nn.W{ly};
net.vW{1} = parameter.nn.vW{ly};
net.dW{1} = parameter.nn.dW{ly};
net.W{2}  = parameter.nn.Ws{ly};
net.vW{2} = parameter.nn.vWs{ly};
net.dW{2} = parameter.nn.dWs{ly};

%% load the data for training
x = parameter.nn.a{ly};
[N,I] = size(x);
kk = randperm(N);
x = x(kk,:);
y = y(kk,:);

%% main loop, train the model
for k = 1 : N
    kp = kp + 1;
    kl = kl + 1;
    
    %% feedforward #1
    net = netffsingle(net, x(k,:), y(k,:));
    
    %% Incremental calculation of x_tail mean and variance
    [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,parameter.nn.a{1}(k,:),kp);
    miu_x_old = miu_x;
    var_x_old = var_x;
    
    %% Expectation of z
    py = probit(miu_x,std_x)';
    for ii = 1:parameter.nn.index
        if ii == parameter.nn.index
            py = sigmf(net.W{1}*py,[1,0]);
        else
            py = sigmf(parameter.nn.W{ii}*py,[1,0]);
        end
        py = [1;py];
        if ii == 1
            Ey2 = py.^2;
        end
    end
    Ey = py;
    Ez = net.W{2}*Ey;
    Ez = exp(Ez);
    Ez = Ez./sum(Ez);
    if parameter.nn.hl > 1
        py = Ey2;
        for ii = 2:parameter.nn.index
            if ii == parameter.nn.index
                py = sigmf(net.W{1}*py,[1,0]);
            else
                py = sigmf(parameter.nn.W{ii}*py,[1,0]);
            end
            py = [1;py];
        end
        Ey2 = py;
    end
    Ez2 = net.W{2}*Ey2;
    Ez2 = exp(Ez2);
    Ez2 = Ez2./sum(Ez2);
    
    %% Network mean calculation
    bias2 = (Ez - y(k,:)').^2;
    ns = bias2;
    NS = norm(ns,'fro');
    
    %% Incremental calculation of NS mean and variance
    [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kp);
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
        grow = 1;
        K = K + 1;
        fprintf('The new node no %d is FORMED around sample %d\n', K, k)
        node(kp)  = K;
        net.W{1}  = [net.W{1};(rand(1,bb) - 0.5)*2*4*sqrt(6/(1 + bb))];
        net.vW{1} = [net.vW{1};zeros(1,bb)];
        net.dW{1} = [net.dW{1};zeros(1,bb)];
        net.W{2}  = [net.W{2} rand(parameter.nn.size(end),1)];
        net.vW{2} = [net.vW{2} zeros(parameter.nn.size(end),1)];
        net.dW{2} = [net.dW{2} zeros(parameter.nn.size(end),1)];
        if ly < parameter.nn.hl
            [wNext,~]             = size(parameter.nn.W{ly+1});
            parameter.nn.W{ly+1}  = [parameter.nn.W{ly+1} (rand(wNext,1) - 0.5)*2*4*sqrt(6/(wNext + 1))];
            parameter.nn.vW{ly+1} = [parameter.nn.vW{ly+1} zeros(wNext,1)];
            parameter.nn.dW{ly+1} = [parameter.nn.dW{ly+1} zeros(wNext,1)];
        end
    else
        grow = 0;
        node(kp) = K;
    end
    
    %% Network variance calculation
    var = Ez2 - Ez.^2;
    NHS = norm(var,'fro');
    
    %% Incremental calculation of NHS mean and variance
    [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kp);
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
    miuminNHS(k,:) = miumin_NHS;
    miustdmin_NHS  = miumin_NHS + (2.6*exp(-NHS)+1.4)*stdmin_NHS;
    VAR(kp,:)      = miu_NHS;
    
    %% pruning hidden unit
    if grow == 0 && K > 1 && miustd_NHS >= miustdmin_NHS && kl > I + 1
        HS = Ey(2:end);
        [~,BB] = min(HS);
        fprintf('The node no %d is PRUNED around sample %d\n', BB, k)
        prune = 1;
        K = K - 1;
        node(kp) = K;
        net.W{1}(BB,:)  = [];
        net.vW{1}(BB,:) = [];
        net.dW{1}(BB,:) = [];
        net.W{2}(:,BB+1)  = [];
        net.vW{2}(:,BB+1) = [];
        net.dW{2}(:,BB+1) = [];
        if ly < parameter.nn.hl
            parameter.nn.W{ly+1}(:,BB+1)  = [];
            parameter.nn.vW{ly+1}(:,BB+1) = [];
            parameter.nn.dW{ly+1}(:,BB+1) = [];
        end
    else
        node(kp) = K;
        prune = 0;
    end
    
    if grow == 1 || prune == 1
        net = netffsingle(net, x(k,:), y(k,:));
    end
    
    %% feedforward #2, executed if there is a hidden node changing
    net = netbackpropagation(net);
    net = netupdate(net);
end

%% substitute the weight back to main model
parameter.nn.W{ly}   = net.W{1};
parameter.nn.vW{ly}  = net.vW{1};
parameter.nn.dW{ly}  = net.dW{1};
parameter.nn.Ws{ly}  = net.W{2};
parameter.nn.vWs{ly} = net.vW{2};
parameter.nn.dWs{ly} = net.dW{2};

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

function nn = netffsingle(nn, x, y)
n = nn.n;
m = size(x,1);
nn.a{1} = x;

%% feedforward from input layer through all the hidden layer
for i = 2 : n-1
    switch nn.activation_function
        case 'sigm'
            nn.a{i} = sigmf(nn.a{i - 1} * nn.W{i - 1}',[1,0]);
        case 'relu'
            nn.a{i} = max(nn.a{i - 1} * nn.W{i - 1}',0);
    end
    nn.a{i} = [ones(m,1) nn.a{i}];
end

%% propagate to the output layer
switch nn.output
    case 'sigm'
        nn.a{n} = sigmf(nn.a{n - 1} * nn.W{n - 1}',[1,0]);
    case 'linear'
        nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
    case 'softmax'
        nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        nn.a{n} = exp( nn.a{n} - max(nn.a{n},[],2));
        nn.a{n} = nn.a{n}./sum(nn.a{n}, 2);
end

%% calculate error
nn.e = y - nn.a{n};

%% calculate loss function
switch nn.output
    case {'sigm', 'linear'}
        nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m;
    case 'softmax'
        nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
end
end

function nn = netbackpropagation(nn)
n = nn.n;
switch nn.output
    case 'sigm'
        d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
    case {'softmax','linear'}
        d{n} = - nn.e;          % dL/dy
end

for i = (n - 1) : -1 : 2
    switch nn.activation_function
        case 'sigm'
            d_act = nn.a{i} .* (1 - nn.a{i}); % contains b
        case 'tanh_opt'
            d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
        case 'relu'
            d_act = zeros(1,length(nn.a{i}));
            d_act(nn.a{i}>0) = 1;
    end
    
    if i+1 == n
        d{i} = (d{i + 1} * nn.W{i}) .* d_act;
    else
        d{i} = (d{i + 1}(:,2:end) * nn.W{i}) .* d_act;
    end
end

for i = 1 : (n - 1)
    if i + 1 == n
        nn.dW{i} = (d{i + 1}' * nn.a{i});
    else
        nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i});
    end
end
end

function nn = netupdate(nn)
for i = 1 : (nn.n - 1)
    dW = nn.dW{i};
    dW = nn.learningRate * dW;
    if(nn.momentum > 0)
        nn.vW{i} = nn.momentum*nn.vW{i} + dW;
        dW = nn.vW{i};
    end
    nn.W{i} = nn.W{i} - dW;
end
end

function [miu,std,var] = meanstditer(miu_old,var_old,x,k)
miu = miu_old + (x - miu_old)./k;
var = var_old + (x - miu_old).*(x - miu);
std = sqrt(var/k);
end

function p = probit(miu,std)
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end

function nn = netconfig(layer)
nn.size                 = layer;
nn.n                    = numel(nn.size);  %  Number of layer
nn.hl                   = nn.n - 2;        %  number of hidden layer
nn.activation_function  = 'sigm';          %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate         = 0.01;            %  learning rate
nn.momentum             = 0.95;            %  Momentum
nn.outputConnect        = 1;               %  1: connect all hidden layer output to output layer, otherwise: only the last hidden layer is connected to output
nn.output               = 'softmax';       %  output layer 'sigm' (=logistic), 'softmax' and 'linear'

%% initiate weights and weight momentum for hidden layer
for i = 2 : nn.n - 1
    nn.W {i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 3 * sqrt(5 / (nn.size(i) + nn.size(i - 1)));
    nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    nn.dW{i - 1} = zeros(size(nn.W{i - 1}));
    nn.c{i - 1} = rand(nn.size(i - 1),1);
end

%% initiate weights and weight momentum for output layer
if nn.outputConnect == 1
    for i = 1 : nn.hl
        nn.Ws {i} = (rand(nn.size(end), nn.size(i + 1)+1) - 0.5) * 2 * 3 * sqrt(5 / (nn.size(end) + nn.size(i + 1)));
        nn.vWs{i} = zeros(size(nn.Ws{i}));
        nn.dWs{i} = zeros(size(nn.Ws{i}));
        nn.beta(i) = 1;
        nn.betaOld(i) = 1;
        nn.p(i) = 1;
    end
else
    nn.Ws  = (rand(nn.size(end), nn.size(end - 1)+1) - 0.5) * 2 * 3 * sqrt(5 / (nn.size(end) + nn.size(end - 1)));
    nn.vWs = zeros(size(nn.Ws));
    nn.dWs = zeros(size(nn.Ws));
end
end

function nn = netconfigtrain(layer)
nn.size   = layer;
nn.n      = numel(nn.size);
nn.activation_function              = 'sigm';       %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate                     = 0.01;  %2      %  learning rate Note:
nn.momentum                         = 0.95;          %  Momentum
nn.output                           = 'softmax';    %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
end

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