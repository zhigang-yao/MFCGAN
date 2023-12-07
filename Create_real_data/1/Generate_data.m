% Specify the desired size of the resized images
targetSize = [64, 64];

% Specify the folder containing the GIF files
folderPath = 'frames';

% Specify the number of times to copy each image
numCopies = 20;

% Specify the standard deviation of the Gaussian noise
noiseStdDev = 0.1;

% Get a list of all the GIF files in the folder
fileList = dir(fullfile(folderPath, '*.gif'));

% Process each GIF file
for i = 1:numel(fileList)
    % Read the current GIF file
    fileName = fullfile(folderPath, fileList(i).name);
    gifData = imread(fileName, 'Frames', 'all');

    % Resize GIF frames to the target size
    numFrames = size(gifData, 4);
    resizedImage = cell(1, numFrames);
    for j = 1:numFrames
        resizedImage{j} = imresize(gifData(:, :, :, j), targetSize);
    end

    % Convert resized frames to grayscale
    grayImage = cell(1, numFrames);
    for j = 1:numFrames
        grayImage{j} = im2gray(resizedImage{j});
    end

    % Copy each gray image and add Gaussian noise
    noisyImages = cell(1, numFrames * numCopies);
    for j = 1:numFrames
        for k = 1:numCopies
            noisyImage = imnoise(grayImage{j}, 'gaussian', 0, noiseStdDev^2);
            noisyImages{(j-1)*numCopies + k} = noisyImage;
        end
    end

    % Save the noisy images as PNG with sequential names
    savepath = 'gray/';
    for j = 1:(numFrames * numCopies)
        saveFileName = fullfile(savepath, [num2str((i-1)*(numFrames * numCopies) + j), '.png']);
        imwrite(noisyImages{j}, saveFileName);
    end
end
