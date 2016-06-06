%% ------------------------------------------------------ %%
function patch = bbCrop2(inputIm, bbs, imPyrd)
%% ------------------------------------------------------ %%
patch = inputIm;
for i=1:size(bbs,1)
	patch(bbs(i,3):bbs(i,4),bbs(i,1):bbs(i,2),:) = imPyrd{i};
end
end