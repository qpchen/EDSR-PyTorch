function Prepare_TrainData_HR_LR_Noise()
%% settings
path_save = './DIV2K';
path_src = './DIV2K/DIV2K_train_HR';
%path_save = './test';
%path_src = './test/HR';
%path_save = './Flickr2K';
%path_src = './Flickr2K/Flickr2K_HR';
ext               =  {'*.jpg','*.png','*.bmp'};
filepaths           =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(path_src, ext{i})));
end
nb_im = length(filepaths);
DIV2K_HR = [];

for idx_im = 1:nb_im
    fprintf('Read HR :%d\n', idx_im);
    ImHR = imread(fullfile(path_src, filepaths(idx_im).name));
    DIV2K_HR{idx_im} = ImHR;
end
%% generate and save LR via imresize() with Bicubic
%sigmas = [10, 30, 50, 70]; % noise level
sigmas = [15, 25, 50]; % noise level
for sigma = sigmas
    for IdxIm = 1:nb_im
        fprintf('IdxIm=%d\n', IdxIm);
        ImHR = DIV2K_HR{IdxIm};
        ImLR = imresize_Noise(ImHR, sigma);
        % name image
        digit = IdxIm;
        fileName = num2str(IdxIm);
        %while digit < 100000  % for Flickr2K
        while digit < 1000
            fileName = ['0', fileName];
            digit = digit*10;
        end
    
        FolderLRx3 = fullfile(path_save, 'DIV2K_train_LR_Noise', ['N', num2str(sigma)]);
        %FolderLRx3 = fullfile(path_save, 'Flickr2K_LR_DN', 'X3');
        
        if ~exist(FolderLRx3)
            mkdir(FolderLRx3)
        end
    
        NameLRx3 = fullfile(FolderLRx3, [fileName, 'n', num2str(sigma), '.png']);
        % save image
        imwrite(ImLR, NameLRx3, 'png');
    end
end

end

function ImLR = imresize_Noise(ImHR, sigma)
% ImLR and ImHR are uint8 data
% downsample by Bicubic
%ImDown = imresize(ImHR, 1/scale, 'bicubic'); % 0-255
ImDown = single(ImHR); % 0-255
ImDownNoise = ImDown + single(sigma*randn(size(ImDown))); % 0-255
ImLR = uint8(ImDownNoise); % 0-255
end