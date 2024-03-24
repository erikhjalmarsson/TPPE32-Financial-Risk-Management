% Reads the data from the quiz to a matrix
data = readmatrix('data.xlsx', 'Sheet', 'Ark1', 'Range', 'A1:A253');
N = 252;
% Question 1
mu = mean(log(data(2:end)./data(1:end-1)));
fprintf("This %f\n", mu);

% Multiplicerar daglig avkastning (mu * 100) med 252 börsdagar
expected = mu * 100 * 252;

fprintf("This %.4f\n", expected);


% Question 2

sigma = sqrt((1/(N-1))*sum( (log(data(2:end)./data(1:end-1)) - mu).^2));
fprintf("Sigma: %.4f", sigma);

expectedSigma = sigma*100*sqrt(252);

fprintf("Expected sigma: %.4f\n", expectedSigma);

% Question 3

lowB = expected - 2*expectedSigma;
upB = expected + 2 * expectedSigma;

% Dividing lowB and upB by 100 to not get the answer in percent
fprintf("The confidence interval is: [%.5f,%.5f]\n", lowB/100, upB/100);

% Question 4 - Skewness

gamma = mean((log(data(2:end)./data(1:end-1)) - mu).^3)/sqrt(mean((log(data(2:end)./data(1:end-1)) - mu).^2))^3;

fprintf("Gamma: %.4f\n", gamma);

% Question 5 - Curtosis

gamma2 = mean((log(data(2:end)./data(1:end-1)) - mu).^4)/sqrt(mean((log(data(2:end)./data(1:end-1)) - mu).^2))^4;

fprintf("Gamma: %.4f\n", gamma2);

% Question 6 - EqWMA 30-28 = 2 30 + 2 = 32 20-29 = 1

for t = 30:252
    var30(t) = (1/30)*sum(log(data(t-28:t+1)./data(t-29:t)).^2);
end

fprintf("Sigma253 EqWMA: %.4f\n", sqrt(var30(252))*100*sqrt(252));

figure(1)
plot(30:252, sqrt(var30(30:252))*100*sqrt(252));
xlim([30 252]);
ylabel('Volatility(%)');
xlabel('30-day moving average');

% Question 7
% The graph indicates the ghost-effect since volatility jumps up/down
% The graph indicated Heteroskedasticity because of periods of high vs low
% volatility
% The graph shows a spike in volatility

% Question 8 - EWMA
lambda = 0.94;
var(1) = (log(data(2)/data(1)))^2;
for t = 2:N
   var(t) = lambda*var(t-1)+(1-lambda)*(log(data(t+1)/data(t)))^2;
end

fprintf("Sigma253 EWMA: %.4f\n", sqrt(var(252))*100*sqrt(252));


% Question 9 - Plot solver lambda vs RiskMetrics

lambdaOpt = 0.8764; % Calculated from solver in excel
MLE(1) = (log(data(2)/data(1)))^2;
for t = 2:N
   MLE(t) = lambdaOpt*MLE(t-1)+(1-lambdaOpt)*(log(data(t+1)/data(t)))^2;
end

% Plot
figure(2)
hold on
plot(1:252, sqrt(MLE)*100*sqrt(252));
plot(1:252, sqrt(var)*100*sqrt(252));
hold off
title('MLE vs RiskMetrics Lambda values');
ylabel('Variance');
legend('MLE, lambda = 0.8764', 'RiskMetrics, lambda = 0.94');
xlim([0 252]);

