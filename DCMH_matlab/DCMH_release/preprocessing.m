function preprocessing()
if ~exist('./data/FLICKR-25K.mat', 'file') || ~exist('./data/vgg_net.mat', 'file')
    cd data;
    fprintf('..load data file and pretrained vgg-f net\n');        
    url = 'http://lamda.nju.edu.cn/jiangqy/data/DCMH_data/data.zip';
    gunzip(url, './');
    unzip('data.zip', './');
    delete('data.zip');
    cd ..;
end

cd matconvnet;
setup;
cd ..;
end