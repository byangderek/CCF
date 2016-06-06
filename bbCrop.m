%% ------------------------------------------------------ %%
function patch = bbCrop(im, bb, meanPixel)
%% ------------------------------------------------------ %%
patch = single(zeros(bb(4)-bb(3)+1,bb(2)-bb(1)+1,3));
patch(:,:,1) = meanPixel(3);
patch(:,:,2) = meanPixel(2);
patch(:,:,3) = meanPixel(1);
[h,w,~] = size(im);
x1 = max(1,2-bb(1));
xx1 = max(1,bb(1));
x2 = x1+min(w,bb(2))-xx1;
xx2 = min(w,bb(2));
y1 = max(1,2-bb(3));
yy1 = max(1,bb(3));
y2 = y1+min(h,bb(4))-yy1;
yy2 = min(h,bb(4));
patch(y1:y2,x1:x2,:) = im(yy1:yy2,xx1:xx2,:);
end