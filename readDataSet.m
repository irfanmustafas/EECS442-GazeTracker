function data = readDataSet(path)
    files = dir(path);
    data = [];
    for f = 3:numel(files)
        im = double(imread(strcat(path, files(f).name)));
        data = [data; im(:)'];
    end
end