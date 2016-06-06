function Pyr = cnnPyramid(I,opts,cnn)
% DEFAULT input params setting:
% input_size [900] must be divideable by stride!
% stride [4]
% pad [16] must be divideable by stride!
% meanPixel [103.939 116.779 123.68] BGR value
%
% You may need to debug codes a little with different 
% input params setting. Practically, stride = 2,4,8 
% all work well.
%

addpath(['path_to_caffe_codes' '/matlab/caffe']);
%caffe('reset');
meanPixel = cnn.meanPix;
input_size = opts.input_size;
stride = opts.stride;
pad = opts.pad;
minDs = opts.minDs;
nPerOct = opts.nPerOct;
nOctUp = opts.nOctUp;
nApprox = opts.nApprox;
if nApprox<0
    nApprox = nPerOct-1;
end
if(minDs<0)
    minDs = min(size(I,1),size(I,2));
end
lambda = opts.lambda;

%% get scales, scalesR, scalesA
[scales, scaleshw] = getScales(I,nPerOct,nOctUp,minDs,stride,1200);
nScales=length(scales); if(1), isR=1; else isR=1+nOctUp*nPerOct; end
isR=isR:nApprox+1:nScales; isA=1:nScales; isA(isR)=[];
j=[0 floor((isR(1:end-1)+isR(2:end))/2) nScales];
isN=1:nScales; for i=1:length(isR), isN(j(i)+1:j(i+1))=isR(i); end
nTypes=0; data=cell(nScales,nTypes); info=struct([]);
imsz=[size(I,1) size(I,2)];

%% compute scalesR imPyrd
flag_cmptd = zeros(nScales,1);
featPyrd = cell(nScales,1);
nR = length(isR);
imPyrd = cell(nR,1);
for i=1:nR
    s=scales(isR(i)); sz1=round(imsz*s/stride)*stride;
    if(all(imsz==sz1)), imPyrd{i}=I; else imPyrd{i}=imResampleMex(I,sz1(1),sz1(2),1); end
end

%% convert imPyrd to cnnFeatPyrd at isR scales
for i=1:nR
    j = isR(i);
    if flag_cmptd(j), continue; end;

    im = imPyrd{i};
    [h,w,~] = size(im);
    if h>=w, flag_lgr=0; else flag_lgr=1; end;
    lgr = max(h,w);
    shr = min(h,w);

    % seg or lay layers of imPyrd to make most use of CNN input_size
    flag_seg = 0;
    flag_lay = 0;
    if lgr>input_size 
        flag_seg = 1; flag_cmptd(j) = 1;
        n0 = ceil(h/input_size);
        n1 = ceil(w/input_size);
        batches = cell(n0*n1,1);
        for p=1:n0
            for q=1:n1
                x1 = (q-1)*input_size-pad+1;
                x2 = q*input_size+pad;
                y1 = (p-1)*input_size-pad+1;
                y2 = p*input_size+pad;
                batches{(p-1)*n1+q} = bbCrop(im,[x1 x2 y1 y2],meanPixel);
            end
        end
    else
        bbs = [pad+1,pad+w,pad+1,pad+h];
        flag_cmptd(j) = 1;
        sz = ceil((lgr+2*pad)/stride)*stride;
        n0 = floor((input_size+2*pad)/sz);
        n=n0;
        if n0>1
            flag_lay = 1;
            n = min(sum(flag_cmptd(isR)==0)+1,n0*n0);
            flag_cmptd(isR(i:i+n-1)) = 1;
            bbs = zeros(n,4);
            for k=1:n
                p = floor((k-1)/n0)+1;
                q = k-(p-1)*n0;
                [h1,w1,~] = size(imPyrd{i+k-1});
                x1 = (q-1)*sz+pad+1;
                x2 = (q-1)*sz+pad+w1;
                y1 = (p-1)*sz+pad+1;
                y2 = (p-1)*sz+pad+h1;
                bbs(k,:) = [x1 x2 y1 y2];
            end
        end
        inputIm = single(zeros(input_size+2*pad,input_size+2*pad,3));
        inputIm(:,:,1) = meanPixel(3);
        inputIm(:,:,2) = meanPixel(2);
        inputIm(:,:,3) = meanPixel(1);
        batches = cell(1,1);
        batches{1} = bbCrop2(inputIm,bbs,imPyrd(i:i+n-1));
    end

    % do CNN forward
    data = prepareBatch(batches,meanPixel);
    feats = cell(length(data),1);
    for k=1:length(data)
        if(~caffe('is_initialized'))
            caffe('init', cnn.model_def, cnn.model_file);
            caffe('set_phase_test');
            caffe('set_device',cnn.device);
            caffe('set_mode_gpu');
        end
        feat = caffe('forward',data(k));
        feat = feat{1};
        feats{k} = permute(feat,[2 1 3]);
    end

    % featPyrd recontruction
    opad = pad/stride;
    oh = ceil(h/stride); ow = ceil(w/stride);
    if flag_seg
        output_size = ceil(input_size/stride);
        layer = single(zeros(oh,ow,size(feats{1},3)));
        k=1;
        for p=1:n0
            for q=1:n1
                feat = feats{k};
                k = k+1;
                x1 = (q-1)*output_size+1;
                x2 = min(ow,q*output_size);
                y1 = (p-1)*output_size+1;
                y2 = min(oh,p*output_size);
                xx1 = opad+1;
                xx2 = min(ow-(q-1)*output_size,output_size)+opad;
                yy1 = opad+1;
                yy2 = min(oh-(p-1)*output_size,output_size)+opad;
                layer(y1:y2,x1:x2,:) = feat(yy1:yy2,xx1:xx2,:);
            end
        end
        featPyrd{j} = layer;
    else
        if flag_lay
            osz = sz/stride;
            feat = feats{1};
            for k=1:n
                [h1,w1,~] = size(imPyrd{i+k-1});
                p = floor((k-1)/n0)+1;
                q = k-(p-1)*n0;
                oh = ceil(h1/stride); ow = ceil(w1/stride);
                xx1 = (q-1)*osz+opad+1;
                xx2 = (q-1)*osz+opad+ow;
                yy1 = (p-1)*osz+opad+1;
                yy2 = (p-1)*osz+opad+oh;
                featPyrd{isR(i+k-1)} = feat(yy1:yy2,xx1:xx2,:);
            end
        else
            featPyrd{j} = feats{1}(opad+1:opad+oh,opad+1:opad+ow,:);
        end
    end
end

%% Approximate cnnFeatPyrd at isA scales
for i=isA
  iR=isN(i); sz1=round(imsz*scales(i)/stride);
  ratio=(scales(i)/scales(iR)).^-lambda;
  featPyrd{i}=imResampleMex(featPyrd{iR},sz1(1),sz1(2),ratio);
end

%% return struct cnnPyrd
Pyr = struct('data',{featPyrd},'nScales',nScales,'scales',scales,'scaleshw',scaleshw);

end


function [scales, scaleshw] = getScales(I,nPerOct,nOctUp,minDs,shrink,sizeLimit)
% set each scale s such that max(abs(round(sz*s/shrink)*shrink-sz*s)) is
% minimized without changing the smaller dim of sz (tricky algebra)
sz=[size(I,1) size(I,2)];
if(any(sz==0)), scales=[]; scaleshw=[]; return; end
nScales = floor(nPerOct*(nOctUp+log2(min(sz./minDs)))+1);
scales = 2.^(-(0:nScales-1)/nPerOct+nOctUp);
if(sz(1)<sz(2)), d0=sz(1); d1=sz(2); else d0=sz(2); d1=sz(1); end
for i=1:nScales, s=scales(i);
  s0=(round(d0*s/shrink)*shrink-.25*shrink)./d0;
  s1=(round(d0*s/shrink)*shrink+.25*shrink)./d0;
  ss=(0:.01:1-eps)*(s1-s0)+s0;
  es0=d0*ss; es0=abs(es0-round(es0/shrink)*shrink);
  es1=d1*ss; es1=abs(es1-round(es1/shrink)*shrink);
  [~,x]=min(max(es0,es1)); scales(i)=ss(x);
end
kp=[scales(1:end-1)~=scales(2:end) true]; scales=scales(kp);
scaleshw = [round(sz(1)*scales/shrink)*shrink/sz(1);
  round(sz(2)*scales/shrink)*shrink/sz(2)]';

nScales = length(scales);
imPyrd = cell(nScales,1);
overLimit = zeros(0,1);
for i=1:length(scales)
  s=scales(i); sz1=round(sz*s/shrink)*shrink;
  if(all(sz1>sizeLimit)), overLimit(end+1) = i; continue; end
end
if(sum(overLimit)>0)
  scales(:,overLimit) = [];
  scaleshw(overLimit,:) = [];
end
end