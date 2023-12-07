% Specify the desired size of the resized images
targetSize = [160, 160];

% Specify the folder containing the GIF files
folderPath = 'frames';

% Specify the number of times to copy each image
numCopies = 20;

% Specify the standard deviation of the Gaussian noise
noiseStdDev = 0.1;

% Get a list of all the GIF files in the folder
fileList = dir(fullfile(folderPath, '*.png'));

count = 0;
% Process each GIF file
for i = 1:numel(fileList)
    % Read the current GIF file
    fileName = fullfile(folderPath, fileList(i).name);
    [gifData,map] = imread(fileName);
    grayImage = ind2gray(gifData,map);

    if mod(i,2) == 0
        savepath = 'test/';
        saveFileName = fullfile(savepath, [num2str(i/2-1), '.png']);
        imwrite(grayImage, saveFileName);
        continue
    end

    % Copy each gray image and add Gaussian noise
    savepath = 'noisy/';
    for j = 1:numCopies
        noisyImage = imnoise(grayImage, 'gaussian', 0, noiseStdDev^2);
        saveFileName = fullfile(savepath, [num2str(count), '.png']);
        imwrite(noisyImage, saveFileName);
        count = count +1;
    end
end
