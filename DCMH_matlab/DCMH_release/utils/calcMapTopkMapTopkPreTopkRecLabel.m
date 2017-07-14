function result = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qB, rB, topk)
%% Function: calcMapTopkMapTopkPreTopkRecLabel:
%   calculate Map, Topk map, Topk precision, Topk recall for hamming ranking task.
% Input:
%   queryLabel: 0-1 label matrix (numQuery * numLabel) for query set.
%   retrievalLabel: 0-1 label matrix (numQuery * numLabel) for retrieval set. 
%   qB: compressed binary code for query set.
%   rB: compressed binary code for retrieval set.
%   topk (optional): vector for different (non-zero and ascending) topk.
% Output:
%   result.map: map for whole retrieval set
%   result.topkMap: vector. topk-Map for different topk
%   result.topkPre: vector. topk-Precision for different topk
%   result.topkRec: vector. topk-Recall for different topk

flag = false;
if exist('topk', 'var')
    flag = true;
end

numQuery = size(qB, 1);
map = 0;
if flag
    nt = numel(topk);
    topkPre = zeros(1, nt);
    topkRec = zeros(1, nt);
    topkMap = zeros(1, nt);
end

for ii = 1: numQuery
    gnd = queryLabel(ii, :) * retrievalLabel' > 0;
    tsum = sum(gnd);
    if tsum == 0
        continue;
    end
    hamm = hammingDist(qB(ii, :), rB);
    [~, index] = sort(hamm);
    gnd = gnd(index);
    count = 1: tsum;
    tindex = find(gnd == 1);
    map = map + mean(count ./ tindex);
    if flag 
        for jj = 1: nt
            tgnd = gnd(1: topk(jj));
            if sum(tgnd) == 0
                continue;
            end
            tcount = 1: sum(tgnd);
            tindex = find(tgnd == 1);
            topkMap(jj) = topkMap(jj) + mean(tcount ./ tindex);
            topkPre(jj) = topkPre(jj) + sum(tgnd) / topk(jj);
            topkRec(jj) = topkRec(jj) + sum(tgnd) / tsum;
        end
    end
end

result.map = map / numQuery;
if flag
    result.topkMap = topkMap / numQuery;
    result.topkPre = topkPre / numQuery;
    result.topkRec = topkRec / numQuery;
end
end
