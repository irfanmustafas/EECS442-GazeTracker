function [ data, label ] = readAllData(dataDir)
    posPath = strcat(dataDir, '/pos/');
    negPath = strcat(dataDir, '/neg/');

    posSet = readDataSet(posPath);
    negSet = readDataSet(negPath);
    
    %negSelection = randperm(size(negSet,1), 5*size(posSet,1));
    %negSet = negSet(negSelection,:);
    
    data = [posSet; negSet];
    label = [ones(size(posSet,1), 1); -1*ones(size(negSet,1), 1)];
end