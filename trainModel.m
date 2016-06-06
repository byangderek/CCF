function trainModel()
% Typically this function consumes MANY memories.
% Running on machines with >128G RAM is recommended
% if you use default training data collected by Acf.
%
% edit the codes of feats loading to fit your case.
%

addpath(genpath('./toolbox-master'));
featDir = 'path_to_feats_of_training_bbs';

X0 = [];
fprintf('loading neg 0...\n');
for i=1:3
    i
    load([featDir 'AcfCaltech+_1up_Is0Stage0_' num2str(i) '.mat']);
    S0 = reshape(feats, [], size(feats,4))';
    X0 = cat(1,X0,S0);
end

fprintf('loading neg 1...\n');
for i=1:3
    i
    load([featDir 'AcfCaltech+_1up_Is0Stage1_' num2str(i) '.mat']);
    S0 = reshape(feats, [], size(feats,4))';
    X0 = cat(1,X0,S0);
end

fprintf('loading neg 2...\n');
for i=1:3
    i
    load([featDir 'AcfCaltech+_1up_Is0Stage2_' num2str(i) '.mat']);
    S0 = reshape(feats, [], size(feats,4))';
    X0 = cat(1,X0,S0);
end

fprintf('loading neg 3...\n');
for i=1:3
    i
    load([featDir 'AcfCaltech+_1up_Is0Stage3_' num2str(i) '.mat']);
    S0 = reshape(feats, [], size(feats,4))';
    X0 = cat(1,X0,S0);
end

clear S0;

X1 = [];
fprintf('loading pos...\n');
for i=1:3
    i
    load([featDir 'AcfCaltech+_1up_Is1Stage0_' num2str(i) '.mat']);
    S1 = reshape(feats, [], size(feats,4))';
    X1 = cat(1,X1,S1);
end
clear S1; clear feats;

detector = struct( 'opts',[], 'clf',[], 'info',[] );
detector.opts = acfTrain();
detector.opts.modelDs = [50 20.5].*2;
detector.opts.modelDsPad = [64 32].*2;
detector.opts.pPyramid.pChns.shrink = 4;
detector.opts.pBoost.nWeak = 4096;
detector.opts.pBoost.discrete = 0;
detector.opts.pBoost.pTree.maxDepth = 5;
detector.opts.pBoost.pTree.nThreads = 16;
detector.opts.pBoost.pTree.fracFtrs = 1/16;
detector.clf = adaBoostTrain(X0,X1,detector.opts.pBoost);
detector.clf.hs = detector.clf.hs + 0.025;
detector.info = 'caltech/vgg_conv3/depth5';
save(['model/Detector_caltech.mat'],'detector');

end