function getFeat_train()
% Running on machine with >32G RAM is recommended,
% or you can change the batch size.
%

addpath(['path_to_caffe_codes' '/matlab/caffe']);
model_def = './data/CaffeNets/VGG_ILSVRC_16_layers_conv3_144.prototxt';
model_file = './data/CaffeNets/VGG_ILSVRC_16_layers.caffemodel';
caffe('reset');
caffe('init', model_def, model_file, 'test');
caffe('set_device',0);
caffe('set_mode_gpu');

mean_pix = [103.939, 116.779, 123.68];

data_path = 'path_to_training_bbs'; % in our case these are generated by training an Acf detector
feat_path = 'path_to_save_feats_of_training_bbs';

% read training bbs
A = dir([data_path '*.mat']);

for l = 1:length(A)
  load([data_path A(l).name]);
  numImg = size(Is,4);

  % pad rectangle to square for caffe input
  Is0 = Is(:,32:-1:1,:,:);
  Is1 = Is(:,end:-1:end-31,:,:);
  Is = cat(2,Is0,Is,Is1);

  % augment bbs as the mini-batch = 100
  flag_aug = 0;
  if mod(numImg,100)~=0
      flag_aug = 1;
      newNum = ceil(numImg/100)*100;
      Is = cat(4,Is,Is(:,:,:,1:(newNum - numImg)));
  end

  % pre-processing data
  Is = {prepare_image(Is, mean_pix)};

  % do caffe forward pass to get feats
  fors = size(Is{1},4)/100;
  feats = cell(fors,1);
  for i=1:fors
      fprintf('%d/%d\n',i,fors);
      train_data = {Is{1}(:,:,:,100*(i-1)+1:100*i)};
      tic;
      feats(i) = caffe('forward', train_data);
      toc;
      feats{i} = permute(feats{i},[2 1 3 4]);
  end
  featss = cat(4,feats{:});
  featss = featss(3:end-2,11:end-10,:,:);
  if flag_aug
      featss = featss(:,:,:,1:numImg);
  end

  % save feats to cache files
  if fors<101
    feats = featss;
    save([feat_path A(l).name],'feats','-v7.3');
  else
    k = floor(fors/100);
    for i=1:k
      feats = featss(:,:,:,(i-1)*10000+1:i*10000);
      save([feat_path A(l).name(1:end-4) '_' num2str(i) '.mat'],'feats','-v7.3');
    end
    if(mod(fors,100))
      feats = featss(:,:,:,k*10000+1:end);
      save([feat_path A(l).name(1:end-4) '_' num2str(k+1) '.mat'],'feats','-v7.3');
    end
  end
end

end

function images = prepare_image(im, mean_pix)
im = single(im);
images = im(:, :, [3 2 1], :);
images = permute(images,[2 1 3 4]);

for c = 1:3
    images(:, :, c, :) = images(:, :, c, :) - mean_pix(c);
end
end