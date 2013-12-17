function [ ret ] = bestOfN( data, N )

data = sortrows(data, 3:5);
uniq = unique(data(:,3).*data(:,4));
ret = [];

for d = 1:numel(uniq)
    curData = data(data(:,3).*data(:,4) == uniq(d), :);
    ret = [ret; curData(1:N,:)];
end

end

