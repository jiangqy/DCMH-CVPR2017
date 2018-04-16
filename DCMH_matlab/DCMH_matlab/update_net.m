function gpu_net = update_net(gpu_net,res_back,lr,N,n_layers,batch_size)
weights_decay = 5*10^-4;
for ii = 1:n_layers
    if isfield(gpu_net.layers{ii},'weights')
        gpu_net.layers{ii}.weights{1} = gpu_net.layers{ii}.weights{1}-...
            lr*(res_back(ii).dzdw{1}/(batch_size*N) + weights_decay*gpu_net.layers{ii}.weights{1});
        gpu_net.layers{ii}.weights{2} = gpu_net.layers{ii}.weights{2}-...
            lr*(res_back(ii).dzdw{2}/(batch_size*N) + weights_decay*gpu_net.layers{ii}.weights{2});
    end
end
end