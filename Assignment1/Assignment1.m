%--------------------------------------------------------------------------
% Title: Financial time series 
%--------------------------------------------------------------------------
% Author: Erik Hjalmarsson, Arvid Johansson
% Date: January 22, 2024
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Task 1
%--------------------------------------------------------------------------

% Import the data
data_weekly = readmatrix('Data.xlsx', 'Sheet', 'Weekly', 'Range', 'A2:C1255');
data_daily = readmatrix('Data.xlsx', 'Sheet', 'Daily', 'Range', 'A2:C6032');

N = 252; 

% Weekly
OMXS30 = data_weekly(:,2);
USDSEK = data_weekly(:,3);
date_w = data_weekly(:,1);
date_w = date_w + 693960;

% Daily
OMXS30_d = data_daily(:,2);
USDSEK_d = data_daily(:,3);
date_d = data_daily(:,1);
date_d = date_d + 693960;

%--------------------------------------------------------------------------
% Subtask a
%--------------------------------------------------------------------------

% Timeseries for OMXS30 & USDSEK
figure(1)
plot(date_w,OMXS30)
ylabel('OMXS30')
yyaxis right %Aktivera den hogra y-axeln
plot(date_w,USDSEK)
ylabel('USD/SEK') %Beskrivning y-axel
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('Datum')
title('Tidsserier') %Titel
legend('OMXS30','USD/SEK','location','northwest')

% Calculating the logarithmic returns for daily and weekly returns
% respectively
Rw_OMXS30 = log(OMXS30(2:end)./OMXS30(1:end-1));
Rw_USDSEK = log(USDSEK(2:end)./USDSEK(1:end-1));
Rd_OMXS30 = log(OMXS30_d(2:end)./OMXS30_d(1:end-1));
Rd_USDSEK = log(USDSEK_d(2:end)./USDSEK_d(1:end-1));

% Logarithmic Returns
figure(2)
plot(date_w(2:end),Rw_OMXS30)
ylabel('OMXS30')
yyaxis right %Aktivera den hogra y-axeln
plot(date_w(2:end),Rw_USDSEK)
ylabel('USD/SEK') %Beskrivning y-axel
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('Datum')
title('Logaritmiska Avkastningar') %Titel
legend('OMXS30','USD/SEK','location','northwest')

output.RIC = {'.OMXS30', 'USDSEK'};

% Average returns OMXS30
mu = mean(Rw_OMXS30);
% Average yearly returns in %
expected = mu * 100 * 52;

% Average returns USDSEK
mu2 = mean(Rw_USDSEK);
% Average yearly returns in %
expected2 = mu2 * 100 * 52;
output.stat.mu = [expected expected2];



% Volatility yearly in % for both respectively
sigma = sqrt((1/(length(Rw_OMXS30)-1))*sum( (Rw_OMXS30 - mu).^2));
expectedSigma = sigma*100*sqrt(52);

sigma2 = sqrt((1/(length(Rw_USDSEK)-1))*sum( (Rw_USDSEK - mu2).^2));
expectedSigma2 = sigma2*100*sqrt(52);

output.stat.sigma = [expectedSigma expectedSigma2];

% 95% Konfidensintervall 
lowB = expected - 2 * expectedSigma;
upB = expected + 2 * expectedSigma;

lowB2 = expected2 - 2 * expectedSigma2;
upB2 = expected2 + 2 * expectedSigma2;

% Dividing lowB and upB by 100 to not get the answer in percent
fprintf("The confidence interval for OMXS30 is: [%.5f,%.5f]\n", lowB/100, upB/100);

fprintf("The confidence interval for USD/SEK is: [%.5f,%.5f]\n", lowB2/100, upB2/100);
output.stat.CI = [lowB/100 upB/100; lowB2/100 upB2/100];

%--------------------------------------------------------------------------
% Subtask b
%--------------------------------------------------------------------------

% Calculate skewness and kurtosis for daily

% Skewness
gamma_daily_OMXS30 = skewness(Rd_OMXS30);
gamma_daily_USDSEK = skewness(Rd_USDSEK);

gamma_weekly_OMXS30 = skewness(Rw_OMXS30);
gamma_weekly_USDSEK = skewness(Rw_USDSEK);

output.stat.skew = [gamma_weekly_OMXS30 gamma_weekly_USDSEK gamma_daily_OMXS30 gamma_daily_USDSEK];

% Kurtosis

gamma2_daily_OMXS30 = kurtosis(Rd_OMXS30) -3;
gamma2_daily_USDSEK = kurtosis(Rd_USDSEK) -3;

gamma2_weekly_OMXS30 = kurtosis(Rw_OMXS30) -3;
gamma2_weekly_USDSEK = kurtosis(Rw_USDSEK) -3;

output.stat.kurt = [gamma2_weekly_OMXS30 gamma2_weekly_USDSEK gamma2_daily_OMXS30 gamma2_daily_USDSEK];



% Histograms

figure(3)
tiledlayout(2,2)
% Top plot
nexttile
histfit(Rw_OMXS30*100)
p_Rw_OMXS30 = percentiles(Rw_OMXS30);
fprintf("%f",p_Rw_OMXS30); 
title('Distribution of log-returns, OMXS30 weekly')
xlabel('log-returns (%)')
ylabel('Frequency')

% Bottom plot
nexttile
histfit(Rw_USDSEK*100)
p_Rw_USDSEK = percentiles(Rw_USDSEK);
title('Distribution of log-returns, USD/SEK weekly')
xlabel('log-returns (%)')
ylabel('Frequency')

nexttile
Rd_OMXS30 = log(OMXS30_d(2:end)./OMXS30_d(1:end-1)); %Elementvis division!
histfit(Rd_OMXS30*100) % Tecken på leptokurtosisk
p_Rd_OMXS30 = percentiles(Rd_OMXS30);
title('Distribution of log-returns, OMXS30 daily')
xlabel('log-returns (%)')
ylabel('Frequency')

nexttile
Rd_USDSEK = log(USDSEK_d(2:end)./USDSEK_d(1:end-1)); %Elementvis division!
histfit(Rd_USDSEK*100);% Tecken på leptokurtosisk
p_Rd_USDSEK = percentiles(Rd_USDSEK);

title('Distribution of log-returns, USD/SEK daily')
xlabel('log-returns (%)')
ylabel('Frequency')

% Aggregating the percentiles in a 4x4
percentiles_total(1,:)=p_Rw_OMXS30;
percentiles_total(2,:)=p_Rw_USDSEK;
percentiles_total(3,:)=p_Rd_OMXS30;
percentiles_total(4,:)=p_Rd_USDSEK;


output.stat.perc = percentiles_total;

% QQ-plot
figure(4)
tiledlayout(2,2)
nexttile
qqplotting(normalize(Rw_OMXS30),'Returns Weekly OMXS30');
nexttile
qqplotting(normalize(Rd_OMXS30),'Returns Daily OMXS30');
nexttile
qqplotting(normalize(Rw_USDSEK),'Returns Weekly USDSEK');
nexttile
qqplotting(normalize(Rd_USDSEK),'Returns Daily USDSEK');

%--------------------------------------------------------------------------
% Task 2
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Subtask a
%--------------------------------------------------------------------------

% EqWMA for OMXS30 weekly 30 day window and 90 day window respectively
volw_OMXS30_30 = EqWMA(30, Rw_OMXS30);
volw_OMXS30_90 = EqWMA(90, Rw_OMXS30);

figure(5)
tiledlayout(2,2)
nexttile
plot(date_w(2:end), volw_OMXS30_30*sqrt(52)*100, 'LineWidth', 1.5, 'Color', 'b');
ylabel('Volatility(%)');
xlabel('30-week moving average');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
title('OMXS30, EqWMA');
nexttile
plot(date_w(2:end), volw_OMXS30_90*sqrt(52)*100, 'LineWidth', 1.5, 'Color', 'r');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('90-week moving average');
title('OMXS30, EqWMA');


% EqWMA for USDSEK
volw_USDSEK_30 = EqWMA(30, Rw_USDSEK);
volw_USDSEK_90 = EqWMA(90, Rw_USDSEK);

nexttile
plot(date_w(2:end), volw_USDSEK_30*sqrt(52)*100, 'LineWidth', 1.5, 'Color', 'b');
ylabel('Volatility(%)');
xlabel('30-week moving average');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
title('USDSEK, EqWMA');
nexttile
plot(date_w(2:end), volw_USDSEK_90*sqrt(52)*100, 'LineWidth', 1.5, 'Color', 'r');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('90-week moving average');
title('USDSEK, EqWMA');

%--------------------------------------------------------------------------
% Subtask b
%--------------------------------------------------------------------------

% EWMA - with RiskMetrics 0.94
volw_OMXS30_RM = EWMA(Rw_OMXS30, 0.94);

% Reads objective and parameters from excel
obj_EWMA_OMXS30 = xlsread('Data.xlsx', 'Weekly', 'I4');
param_EWMA_OMXS30 = xlsread('Data.xlsx', 'Weekly', 'I2');
obj_EWMA_USDSEK = xlsread('Data.xlsx', 'Weekly', 'P4');
param_EWMA_USDSEK = xlsread('Data.xlsx', 'Weekly', 'P2');

output.EWMA.obj = [obj_EWMA_OMXS30 obj_EWMA_USDSEK];
output.EWMA.param = [param_EWMA_OMXS30 param_EWMA_USDSEK];

figure(6)
tiledlayout(1,2)

% USDSEK EWMA
volw_USDSEK_RM = EWMA(Rw_USDSEK, 0.94);
nexttile
plot(date_w(2:end), volw_USDSEK_RM*sqrt(52)*100, 'LineWidth', 0.5, 'Color', 'b');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('Lambda 0.94');
title('USDSEK, EWMA');

% OMXS30 EWMA
nexttile
plot(date_w(2:end), volw_OMXS30_RM*sqrt(52)*100, 'Color', 'r');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('Lambda 0.94');
title('OMXS30, EWMA');

%--------------------------------------------------------------------------
% Subtask c
%--------------------------------------------------------------------------

% EWMA & GARCH(1,1) optimal lambda model
% Optimal lambda for EWMA OMXS30 using Excel solver = 0,91335113
% Optimal lambda for EWMA USDSEK using Excel solver = 0,902396544
% Optimal long run volatility estimated from GARCH(1,1): 0,015600392

volw_USDSEK = EWMA(Rw_USDSEK, 0.902396544);
volw_OMXS30 = EWMA(Rw_OMXS30, 0.91335113);

% USDSEK with estimated optimal lambda
figure(7)
tiledlayout(1,2)
nexttile
plot(date_w(2:end), volw_USDSEK*sqrt(52)*100, 'LineWidth', 0.5, 'Color', 'b');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('USDSEK Estimated Lambda');
title('USDSEK, EWMA');

% OMXS30 with estimated optial lambda
nexttile
plot(date_w(2:end), volw_OMXS30*sqrt(52)*100, 'LineWidth', 0.5, 'Color', 'r');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln
xlabel('OMXS30 Estimated Lambda');
title('OMXS30, EWMA');

%%%%%%%%%%%%%%%%%%%%%%
% Variance_Targeting %
%%%%%%%%%%%%%%%%%%%%%%
% OMXS30
% obj_GARCHVT = readmatrix('Data.xlsx', 'Sheet', 'Variance_Targeting','Range', 'Z5');
obj_GARCHVT = xlsread('Data.xlsx', 'Variance_Targeting', 'Z5');
param_garchVT = readmatrix('Data.xlsx', 'Sheet', 'Variance_Targeting', 'Range', 'Y21:AA21');

% USDSEK
% obj_GARCH2VT = readmatrix('Data.xlsx', 'Sheet', 'Variance_Targeting','Range', 'M5');
obj_GARCH2VT = xlsread('Data.xlsx', 'Variance_Targeting', 'M5');

output.GARCH.objVT = [obj_GARCHVT obj_GARCH2VT];
param_garch2VT = readmatrix('Data.xlsx', 'Sheet', 'Variance_Targeting', 'Range', 'L21:N21');


output.GARCH.paramVT = [param_garchVT param_garch2VT];

%--------------------------------------------------------------------------
% Subtask d
%--------------------------------------------------------------------------

% Importing GARCH from Excel (Variance)

% USDSEK
GARCH_data = readmatrix('Data.xlsx', 'Sheet', 'GARCH', 'Range', 'G4:G1255');
% OMXS30
GARCH_data2 = readmatrix('Data.xlsx', 'Sheet', 'GARCH', 'Range', 'T4:T1255');


figure(8)
tiledlayout(1,2)
nexttile
plot(date_w(3:end), sqrt(GARCH_data)*sqrt(52)*100, 'LineWidth', 0.5, 'Color', 'b');
title('GARCH USDSEK Weekly');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln


nexttile
plot(date_w(3:end), sqrt(GARCH_data2)*sqrt(52)*100, 'LineWidth', 0.5, 'Color', 'r');
title('GARCH OMXS30 Weekly');
ylabel('Volatility(%)');
datetick('x','yy') %Satter datumformatet yy pa x-axeln


% OMXS30
obj_GARCH = xlsread('Data.xlsx', 'GARCH', 'Z5');
param_garch = readmatrix('Data.xlsx', 'Sheet', 'GARCH', 'Range', 'Y21:AA21');

% USDSEK
obj_GARCH2 = xlsread('Data.xlsx', 'GARCH', 'M5');
output.GARCH.obj = [obj_GARCH obj_GARCH2];
param_garch2 = readmatrix('Data.xlsx', 'Sheet', 'GARCH', 'Range', 'L21:N21');


output.GARCH.param = [param_garch param_garch2];



% First return is not accounted for  
standard_garch2 = Rw_OMXS30(2:end)./sqrt(GARCH_data2);
standard_garch = Rw_USDSEK(2:end)./sqrt(GARCH_data);

%GARCH_data2 = OMXS30
%GARCH_data = USDSEK

% QQ-plots
figure(9)
tiledlayout(1,2)
nexttile
qqplotting(standard_garch, 'USDSEK');
nexttile
qqplotting(standard_garch2, 'OMXS30');


%--------------------------------------------------------------------------
% Task 3
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Subtask a
%--------------------------------------------------------------------------

correlation = corr(Rw_OMXS30, Rw_USDSEK);
output.stat.corr = correlation;

%--------------------------------------------------------------------------
% Subtask b
%--------------------------------------------------------------------------

[acf, lags] = autocorr(Rw_OMXS30);
[acf2, lags2] = autocorr(Rw_USDSEK);

output.stat.acorr = [acf(2:6,:) acf2(2:6,:)];

%--------------------------------------------------------------------------
% Subtask c
%--------------------------------------------------------------------------

returnsWeeklyOMXS30 = Rw_OMXS30;
returnsWeeklyUSDSEK = Rw_USDSEK;

% Extracts the epsilon value using the estimated volatility and returns 
% Formula: R_n = Sigma_n*Epsilon_n ==> Epsilon_n = R_n/Sigma_n. Remember
% GARCH_data is the variance for each n
% The first return is not accounted for
% IFM
epsilon_USDSEK = Rw_USDSEK(2:end)./sqrt(GARCH_data);
epsilon_OMXS30 = Rw_OMXS30(2:end)./sqrt(GARCH_data2);

% Transform epsilon to the unit domain [0 1]
trans_epsilon_USDSEK = normcdf(epsilon_USDSEK);
trans_epsilon_OMXS30 = normcdf(epsilon_OMXS30);
joint_data = [trans_epsilon_USDSEK trans_epsilon_OMXS30];

% Five types of distributions
gaussian = copulafit('gaussian', joint_data);
clayton = copulafit('clayton', joint_data);
frank = copulafit('frank', joint_data);
% Also returns the degrees of freedom for the student t distribution
[student_t, degreef] = copulafit('t', joint_data);
gumbel = copulafit('gumbel', joint_data);

% Log-likelihood values for the different copulas 
log_gaussian = sum(log(copulapdf('gaussian', joint_data, gaussian)));
log_clayton = sum(log(copulapdf('clayton', joint_data, clayton)));
log_frank = sum(log(copulapdf('frank', joint_data, frank)));
log_student_t = sum(log(copulapdf('t', joint_data, student_t, degreef)));
log_gumbel = sum(log(copulapdf('gumbel', joint_data, gumbel)));

fprintf(" \n Gaussian: %.2f\n Clayton: %.2f\n Frank: %.2f\n Student_t: %.2f\n Gumbel: %.2f\n", log_gaussian, log_clayton, log_frank, log_student_t, log_gumbel);

output.copulaLogL = [log_gaussian log_student_t log_gumbel log_clayton log_frank];

% Student t distribution is the best because it yields the highest
% log-likelihood

% 1253 - size of dataset
generated_data = copularnd('t', student_t, degreef, 1253);

% Scatter of generated data
figure(10)
tiledlayout(1,2)
nexttile
scatter(generated_data(:,1), generated_data(:, 2), 'x');
title('Randomly generated data from the copula');

% Scatter plot of historical returns
nexttile
scatter(joint_data(:,1), joint_data(:,2), 'x');
title('Historical data');

% The patterns observed in the scatter plots are at first sight very
% similar. However, it is evident that the simulated data cant properly
% predict outliers in the historical data.

% Just for fun, this shows us that there is little to no correlation
% between the returns
figure(12)
scatter(Rw_OMXS30, Rw_USDSEK);
title('Scatter plots of the returns from USDSEK and OMXS30');

printResults(output, true);

%--------------------------------------------------------------------------
% Functions
%--------------------------------------------------------------------------

function minusloglikelihood = loglikelihood(returns, variance)
    minusloglikelihood = sum(-log(variance)-(returns^2)/variance);
end

function volatility = EWMA(returns, lambda)
    var(1) = sqrt((returns(1))^2);
    for t = 2:length(returns)
       var(t) = lambda*var(t-1)^2+(1-lambda)*(returns(t))^2;
    end
    volatility = sqrt(var);
end

function var = EqWMA(windowSize, dist)
    for t = windowSize:length(dist)
        volatility(t) = sqrt((1/windowSize)*sum(dist(t-windowSize+1:t).^2));
    end
    var = volatility;
end

% Own implementation of qqplot
function qqplotting(dist, name)
    n = length(dist);
    sorted = sort(dist);
    u_qq = norminv(((1:n)-0.5)/n, 0, 1);
    
    scatter(u_qq, sorted);
    ylabel('Quantiles of input Sample');
    xlabel('Standard Normal Quantiles');
    line_lim = [min(u_qq), max(u_qq)];
    hold on;
    plot(line_lim, line_lim, 'Color', 'red', 'LineStyle', '-.');
    title(name);
    hold off;
end

function per = percentiles(dist)
    per(1) = prctile(dist, 1)*100;
    per(2) = prctile(dist, 5)*100;
    per(3) = prctile(dist, 95)*100;
    per(4) = prctile(dist, 99)*100;

end

function normalize = norming(dist) 
    mean__ = mean(dist);
    volatility = std(dist);
    normalize = (dist - mean__)/volatility;
end

% Decided to use matlabs function for skewness and kurtosis.
%{
function gamma = skewness(dist)
    mu_ = mean(log(dist(2:end)./dist(1:end-1)));
    gamma = mean((log(dist(2:end)./dist(1:end-1)) - mu_).^3)/sqrt(mean((log(dist(2:end)./dist(1:end-1)) - mu_).^2))^3;
end

function gamma2 = kurtosis(dist)
    mu_ = mean(log(dist(2:end)./dist(1:end-1)));
    temp = mean((log(dist(2:end)./dist(1:end-1)) - mu_).^4)/sqrt(mean((log(dist(2:end)./dist(1:end-1)) - mu_).^2))^4;
    gamma2 = temp - 3;
end
%}


