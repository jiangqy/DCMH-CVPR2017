function result = process_DCMH(dataset, param)
XTrain = dataset.X(:, :, :, param.indexTrain);
YTrain = dataset.Y(param.indexTrain, :);
trainLabel = dataset.L01(param.indexTrain,:);

S = trainLabel * trainLabel' > 0;

bit = param.bit;
gamma = param.gamma;
eta = param.eta;

lr_img = param.lr_img;
lr_txt = param.lr_txt;

[num_train, dy] = size(YTrain);

maxIter = param.maxIter;
F = zeros(bit, num_train);
G = zeros(bit, num_train);
Y = YTrain';

load('./data/vgg_net.mat');

txt_net = net_structure_txt(dy, bit);
img_net = net_structure_img(net, bit);

loss = zeros(1, maxIter);

batch_size = param.batch_size;
for epoch = 1: maxIter
    B = e_sign(gamma*(F + G));
    
    for ii = 1: ceil(num_train/ batch_size)
        R = randperm(num_train);
        index = R(1: batch_size);
        y = Y(:, index);
        y = gpuArray(single(reshape(y,[1,size(y,1),1,size(y,2)])));
        res = vl_simplenn(txt_net,y);
        output = gather(squeeze(res(end).x));
        G(:,index) = output;
        
        dJdLogloss = 0.5*F*(1 ./ (1+exp(-0.5*F'*G(:,index)))-S(:,index));        
        G1 = G*ones(num_train,1);
        dJdGB = 2*gamma *(G(:,index)-B(:,index))+2*eta*repmat(G1,1,numel(index));
        dJdGb = dJdLogloss + dJdGB;
        dJdGb = single(gpuArray(reshape(dJdGb,[1,1,size(dJdGb,1),size(dJdGb,2)])));
        res = vl_simplenn(txt_net,y,dJdGb);            

        n_layers = numel(txt_net.layers);
        txt_net = update_net(txt_net,res,lr_txt(epoch),num_train,n_layers,batch_size);

    end
    
    for ii = 1:ceil(num_train/batch_size)
        R = randperm(num_train);
        index = R(1:batch_size);
        img = single(XTrain(:,:,:,index));
        im_ = img - repmat(net.meta.normalization.averageImage,1,1,1,size(img,4));
        im_ = gpuArray(im_);
        
        res = vl_simplenn(img_net,im_);
        n_layers = numel(img_net.layers);
        output = gather(squeeze(res(end).x));
        F(:,index) = output;
        
        F1 = F*ones(num_train,1);
        dJdB = 2*gamma*(F(:,index)-B(:,index)) + 2*eta*repmat(F1,1,numel(index));
        dJdLogloss = 0.5*G*(1 ./ (1+exp(-0.5*G'*F(:,index))) - S(:,index));
        dJdFb = dJdLogloss + dJdB;
        dJdFb = reshape(dJdFb,[1,1,size(dJdFb,1),size(dJdFb,2)]);
        dJdFb = gpuArray(single(dJdFb));
        
        res = vl_simplenn(img_net,im_, dJdFb);
        img_net = update_net(img_net,res,lr_img(epoch),num_train,n_layers,batch_size);
    end
    
    l = calc_loss(S,F,G,B,gamma,eta,num_train);
    fprintf('...epoch: %3d/%d\tloss:%3.3f\n',epoch,maxIter,...
        l);
    loss(epoch) = l;    
end
fprintf('...training finishes\n');
XRetrieval = dataset.X(:,:,:,param.indexRetrieval);
YRetrieval = dataset.Y(param.indexRetrieval,:);
retrievalLabel = dataset.L01(param.indexRetrieval,:);

XQuery = dataset.X(:,:,:,param.indexQuery);
YQuery = dataset.Y(param.indexQuery,:);
queryLabel = dataset.L01(param.indexQuery,:);

[rBX] = generateImgCode(img_net,XRetrieval,bit);
[qBX] = generateImgCode(img_net,XQuery,bit);
[rBY] = generateTxtCode(txt_net,YRetrieval',bit);
[qBY] = generateTxtCode(txt_net,YQuery',bit);

rBX = compactbit(rBX > 0);
rBY = compactbit(rBY > 0);
qBX = compactbit(qBX > 0);
qBY = compactbit(qBY > 0);
fprintf('...encoding finishes\n');
result.rBX = rBX;
result.rBY = rBY;
result.qBX = qBX;
result.qBY = qBY;

% hamming ranking
result.hri2t = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qBX, rBY);
result.hrt2i = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qBY, rBX);

% hash lookup
result.hli2t = calcPreRecRadiusLabel(queryLabel, retrievalLabel, qBX, rBY);
result.hlt2i = calcPreRecRadiusLabel(queryLabel, retrievalLabel, qBY, rBX);


result.loss = loss;
end

function l = calc_loss(S,F,G,B,gamma,eta,num_train)
theta = 0.5*F'*G;
Logloss = log(1+exp(theta))-theta.*S;
l = sum(Logloss(:))+gamma*(norm(F-B,'fro')^2+norm(G-B,'fro')^2)+...
            eta*(norm(F*ones(num_train,1),'fro')^2+norm(G*ones(num_train,1),'fro')^2);

end

function B = generateImgCode(img_net,images,bit)
batch_size = 256;
num = size(images,4);
B = zeros(num,bit);
for i = 1:ceil(num/batch_size)
    index = (i-1)*batch_size+1:min(i*batch_size,num);
    image = single(images(:,:,:,index));
    im_ = imresize(image,img_net.meta.normalization.imageSize(1:2));
    im_ = im_ - repmat(img_net.meta.normalization.averageImage,1,1,1,size(im_,4));        
    res = vl_simplenn(img_net,gpuArray(im_));
    output = gather(squeeze(res(end).x));
    B(index,:) = sign(output');
end
end

function B = generateTxtCode(txt_net,text,bit)
num = size(text,2);
batch_size = 5000;
B = zeros(num,bit);
for i = 1:ceil(num/batch_size)
    index = (i-1)*batch_size+1:min(i*batch_size,num);
    y = text(:,index);
    y = gpuArray(single(reshape(y,[1,size(y,1),1,size(y,2)])));
    res = vl_simplenn(txt_net,y);
    output = gather(squeeze(res(end).x));
    B(index,:) = sign(output');
end


end

function B = e_sign(A)
B = zeros(size(A));
B(A > 0) = 1;
B(A < 0) = -1;
end