load('face/channel_mean.mat');

fs = fs(randperm(300,100),:);

load P_face;
ls = chnsScaling( P.scales', fs, 1 );
ls = round(ls*10^5)/10^5;
lambda = ls
%saveas(gcf,'face/powerlaw_face','fig');