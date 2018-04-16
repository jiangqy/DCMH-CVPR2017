function net = net_structure_img(net,codelens)
n = numel(net.layers);

for i = 1:n
    if isfield(net.layers{i},'weights')
        net.layers{i}.weights{1} = gpuArray(net.layers{i}.weights{1});
        net.layers{i}.weights{2} = gpuArray(net.layers{i}.weights{2});
    end
end

net.layers{20}.pad = [0,0,0,0];
net.layers{20}.stride = [1,1];
net.layers{20}.type = 'conv';
net.layers{20}.name = 'fc8';
net.layers{20}.weights{1} = gpuArray(0.01*randn(1,1,4096,codelens,'single'));
net.layers{20}.weights{2} = gpuArray(0.01*randn(1,codelens,'single'));
net.layers{20}.opts = {};

end