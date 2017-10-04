load('C:\Users\Daller\Dropbox\Kandidat i Fysik-Geofysik\Master Thesis\matlab\IMPALA_CPU\list100.mat')
D = cell2mat(list(:,1));

listLength = size(D,1);

matchVec = zeros(listLength,1);

dataEventNumber = randi([1,listLength]);

dataEvent = D(dataEventNumber,:);



k = parallel.gpu.CUDAKernel('cudaFind.ptx','cudaFind.cu');