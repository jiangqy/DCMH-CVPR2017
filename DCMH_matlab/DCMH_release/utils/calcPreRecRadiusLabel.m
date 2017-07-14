function result = calcPreRecRadiusLabel(queryLabel, retrievalLabel, qB, rB)
%% Function: calcPreRecRadiusLabel
%   calculate precision and recall within different radius based on Label.
% Input:
%   queryLabel: 0-1 label matrix (numQuery * numLabel) for query set.
%   retrievalLabel: 0-1 label matrix (numQuery * numLabel) for retrieval set. 
%   qB: compressed binary code for query set.
%   rB: compressed binary code for retrieval set.
% Output:
%   result.Pre: maxR-dims vector. Precision within different hamming radius.
%   result.Rec: maxR-dims vector. Recall within different hamming radius.
Wtrue = queryLabel * retrievalLabel' > 0;
Dhamm = hammingDist(qB, rB);

maxHamm = max(Dhamm(:));
totalGoodPairs = sum(Wtrue(:));

% find pairs with similar codes
precision = zeros(maxHamm, 1);
recall = zeros(maxHamm, 1);
for n = 1: length(precision)
    j = (Dhamm <= ((n-1) + 00.001));
    retrievalGoodPairs = sum(Wtrue(j));
    
    retrievalPairs = sum(j(:));
    precision(n) = retrievalGoodPairs / (retrievalPairs + eps);
    recall(n) = retrievalGoodPairs / totalGoodPairs;
end

result.Pre = precision;
result.Rec = recall;
end
