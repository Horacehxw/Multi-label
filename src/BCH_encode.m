load("../.temp/train/y_bar.mat");
y_bars = single(y_bars);
[iter, N, L_bar] = size(y_bars);
L_tilde = double(L_tilde);

y_tildes = zeros([iter, N, L_tilde], 'single'); % numbers
for ii=1:iter
    y_bar_ii = reshape(y_bars(ii, :, :), [N, L_bar]);
    temp = bchenc(gf(y_bar_ii), L_tilde, L_bar);
    y_tildes(ii,:,:) = temp.x;
end

[N_, L_bar] = size(y_test);
temp = bchenc(gf(y_test), L_tilde, L_bar);
y_test_tilde = temp.x;


save("../.temp/train/y_tilde.mat", "y_tildes", "y_test_tilde");

