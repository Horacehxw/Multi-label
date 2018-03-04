load("../.temp/test/y_tilde_hat.mat");

y_tilde_hats = single(y_tilde_hats);
[iter, N, L_tilde] = size(y_tilde_hats);
L_bar = double(L_bar);

y_bar_hats = zeros([iter, N, L_bar], 'single');
for ii = 1:iter
    y_tilde_hat_ii = reshape(y_tilde_hats(ii, :, :), [N, L_tilde]);
    temp = bchdec(gf(y_tilde_hat_ii), L_tilde, L_bar);
    y_bar_hats(ii, :, :) = temp.x;
end

save("../.temp/test/y_bar_hat.mat", "y_bar_hats")
