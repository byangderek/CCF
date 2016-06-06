%% ------------------------------------------------------ %%
function data = prepareBatch(batches, meanPixel)
	% mean-subtraction, permute...
%% ------------------------------------------------------ %%
nBatch = length(batches);
data = cell(nBatch,1);
for i=1:nBatch
	im = batches{i}(:,:,[3 2 1]);
	im = permute(im,[2 1 3]);
	for c=1:3
		im(:,:,c) = im(:,:,c)-meanPixel(c);
	end
	data{i} = im;
end
end