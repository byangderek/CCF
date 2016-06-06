function allBBs = cnnDetect(fs, ds, opts, cnn)

testNum = length(fs);
nDs = length(ds);
allBBs = cell(testNum,1);
for i=1:testNum
    fprintf('\n%d/%d\n',i,testNum);
	I = imread(fs{i});
    w = size(I,2);
    I = single(I);
    if(size(I,3)==1)
        I = cat(3,I,I,I);
    end
    if(opts.imresize~=1)
        I = imresize(I,opts.imresize);
    end
    if(opts.imflip)
        I = I(:,end:-1:1,:);
    end
    
    posIds = find(fs{i}=='/');
    imgNm = fs{i}(posIds(end)+1:end);
    tic;
	bb = cnnDetectImg(I,[],ds,opts,cnn,imgNm);
    toc;
    if(isempty(bb))
        bb = zeros(0,6);
    else
        if(opts.imresize~=1)
            bb(:,1:4) = bb(:,1:4)./opts.imresize;
        end
    end
    allBBs{i} = bb;
end

end



function bbs = cnnDetectImg( I, P, ds, opts, cnn, imgNm )
	
stride = opts.stride;
modelDsPad = ds{1}.opts.modelDsPad;
modelDs = ds{1}.opts.modelDs;
shrink = opts.stride;
cascThr = -1;
pad = (modelDsPad-modelDs)./(shrink*2);
pyrdNm = ['path_to_save_pyrd' imgNm '.mat'];
if opts.savePyrd && exist(pyrdNm,'file')
    load(pyrdNm);
else
    if isempty(P)
        P = cnnPyramid(I,opts,cnn);
    end
    if(opts.savePyrd)
        save(['path_to_save_pyrd' imgNm '.mat'],'P');
    end
end

if(opts.addCf)
    cfOpts = ds{1}.cfopts;
    sz = size(I);
    for i=1:P.nScales
        I1 = imresize(I,P.scaleshw(i,:).*sz(1:2));
        I1 = I1./max(255,max(I1(:)));
        C=chnsCompute(I1,cfOpts.pPyramid.pChns);
        C=convTri(cat(3,C.data{:}),cfOpts.pPyramid.smooth);
        P.data{i} = cat(3,P.data{i},C);
    end
end

nDs = length(ds);
bbs = cell(P.nScales,nDs);
for i=1:P.nScales
  for j=1:nDs
  	layer = imPad(P.data{i},pad,'replicate');
    bb = acfDetect1(layer,ds{j}.clf,shrink,...
      modelDsPad(1),modelDsPad(2),stride,cascThr);
    shift=(modelDsPad-modelDs)/2-pad;
    bb(:,1)=(bb(:,1)+shift(2))/P.scaleshw(i,2);
    bb(:,2)=(bb(:,2)+shift(1))/P.scaleshw(i,1);
    bb(:,3)=modelDs(2)/P.scales(i);
    bb(:,4)=modelDs(1)/P.scales(i);
    if(nDs>1)
        bb(:,6)=j;
    end
    bbs{i,j}=bb;
  end
end; 
bbs=cat(1,bbs{:});
bbs=bbNms(bbs,ds{1}.opts.pNms);
end
