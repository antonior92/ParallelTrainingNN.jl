% Download data
% Training set
M = csvread('processed_data/training_set.csv', 2);
ti = M(1:end, 1);
ui = M(1:end, 2);
yi = M(1:end, 3);
% Test set
M = csvread('processed_data/test_set.csv', 2);
tv = M(1:end, 1);
uv = M(1:end, 2);
yv = M(1:end, 3);
% Create structures
Zi = iddata(yi, ui, ti(2)-ti(1));
Zv = iddata(yv, uv, tv(2)-tv(1));


% Identification
[Zi, Ti] = detrend(Zi);
sys = arx(Zi, [1, 1, 1]);
%Zi = retrend(Zi, Ti);

% Compare on the training set
[Zv, Tv] = detrend(Zv);
[yh, fit, x0] = compare(Zv, sys);
Zv = retrend(Zv, Tv);
yh = retrend(yh, Tv);

mse = goodnessOfFit(yh.y, Zv.y, 'MSE')