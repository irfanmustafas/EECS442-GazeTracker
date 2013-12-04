function [svm] = trainSvm(dataDir)
    [data, label] = readAllData(dataDir);
    svm = svmtrain(data, label); %, 'kernel_function', 'rbf');
    fd = fopen('svm.py', 'w');
    fprintf(fd, 'import numpy\n\n');
    fprintf(fd, 'w = numpy.array([');
    fprintf(fd, '%20.10f,', svm.Alpha'*svm.SupportVectors);
    fprintf(fd, '], dtype=numpy.double)\n');
    
    fprintf(fd, 'bias = %20.10f\n', svm.Bias);
    
    fprintf(fd, 'scale = numpy.array([');
    fprintf(fd, '%20.10f,', svm.ScaleData.scaleFactor);
    fprintf(fd, '], dtype=numpy.double)\n');
    
    fprintf(fd, 'shift = numpy.array([');
    fprintf(fd, '%20.10f,', svm.ScaleData.shift);
    fprintf(fd, '], dtype=numpy.double)\n');
    
    fseek(fd, 0, 'eof');
end

