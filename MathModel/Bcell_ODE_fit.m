%(c) 2020 Signaling Systems Lab UCLA
%All rights reserved. 
%This MATLAB code implements the mathematical modeling on the B-cell population
%dynamics. It fitted the birth-death ODE to the cell number measurements
%and inferred the differential rates.

%A detailed application of this package is given in the main text. 

%% Specify the cell population names
clear all;
filefolder=pwd;
Name={'Fr.A','Fr.B','Fr.C','Fr.C''','Fr.D','Fr.E'};
Fractions = ["Fraction A", "Fraction B", "Fraction C", "Fraction C'", "Fraction D", "Fraction E"];
SpecieNum=size(Name,2);

%% Load data: cell number measurements, and proliferation, death rates
color=jet(SpecieNum);
data_filename = 'Expt_Data_for_Model_Fitting.xlsx';
conditions = 'KOvsWT'; %'KOvsWT'; %'OldvsYoung'
samples = ["Young", "OldN", "OldS"]; %["WT", "IkB-", "IkB-RelA+/-"];
GRType = 2; %corresponds to net proliferation rate scenario

if strmatch(conditions,'KOvsWT')
    [WT, ~]= xlsread(data_filename,1,'L3:Q14');
    [KO, ~]= xlsread(data_filename,1,'L16:Q27');
    [KO2, ~]= xlsread(data_filename,1,'L29:Q36');
    SampleNames = [("WT_"+[1:size(WT, 1)]'); ("IkB-_" +[1:size(KO, 1)]'); ("IkB-RelA+/-_" +[1:size(KO2, 1)]')];
else
    [WT, ~]= xlsread(data_filename,2,'L3:Q11');
    [KO, ~]= xlsread(data_filename,2,'L13:Q17');
    [KO2, ~]= xlsread(data_filename,2,'L19:Q23');
    SampleNames = [("Young_"+[1:size(WT, 1)]'); ("OldN_" +[1:size(KO, 1)]'); ("OldS_" +[1:size(KO2, 1)]')];
end

%% Input the proliferation and death rates
[GR_WT, ~]= xlsread(data_filename,GRType+2,'B2:G2');
[GR_KO, ~]= xlsread(data_filename,GRType+2,'B3:G3');

DifferentialRateSummary=[];TheoryCellNumberSummary=[];
FluxX_ReplicateSummary=[];FluxY_ReplicateSummary=[];
ResidenceTimeSummary=[];
mean_DiffRate = [];
std_DiffRate = [];

for DataType=1:3 %For the 3 expt conditions

if DataType==1
    Data=WT; figurename=[filefolder,'\WT'];
    GR=GR_WT;
elseif  DataType==2
    Data=KO; figurename=[filefolder,'\KO'];
    GR=GR_KO;
elseif  DataType==3
    Data=KO2; figurename=[filefolder,'\KO2'];
    GR=GR_KO;
end

Mean=mean(Data,1);
Variance=var(Data,1);
Sd=std(Data,1);
Xaxis=1:1:size(Data,2);
yLimHigh= 20;

%% Fit differential rates for each replicate and export data

Frac1=1;
Frac2=0;
TheoryCellNumber=Data;
DifferentialRateReplicate=Data;
ResidenceTimeReplicate=Data;
FluxX_Replicate=Data;
FluxY_Replicate=Data;
for kk=1:size(Data,1)
DifferentiaRate=1;%Input the first one
Theory2=Data(kk,1);
SumTemp=0;
for i=2:size(GR,2) 
    DifferentiaRate=[DifferentiaRate DifferentiaRate(end)*Data(kk,i-1)/Data(kk,i)+GR(i)];
    Frac1=[Frac1,Data(kk,1)/Data(kk,i)];
    Frac2=[Frac2,sum(GR(2:i).*Data(kk,2:i))/Data(kk,i)];
    if DifferentiaRate(i)-GR(i)~=0
        Theory2=[Theory2,DifferentiaRate(i-1)/(DifferentiaRate(i)-GR(i))*Theory2(end)];
    else
        display('singularity');
    end
  
end
TheoryCellNumber(kk,:)=Theory2;
DifferentialRateReplicate(kk,:)=DifferentiaRate;
ResidenceTimeReplicate(kk,:)=1./(DifferentiaRate-GR);
FluxX_Replicate(kk,:)= cumsum(1./DifferentiaRate);
FluxY_Replicate(kk,:)= cumsum(GR.*Theory2)/Theory2(1);
end

TheoryCellNumberSummary=[TheoryCellNumberSummary;TheoryCellNumber];

DifferentialRateSummary=[DifferentialRateSummary;DifferentialRateReplicate];
mean_DiffRate = [mean_DiffRate; mean(DifferentialRateReplicate, 1)];
std_DiffRate = [std_DiffRate; std(DifferentialRateReplicate, 1)];

ResidenceTimeSummary=[ResidenceTimeSummary;ResidenceTimeReplicate];

FluxX_ReplicateSummary=[FluxX_ReplicateSummary;FluxX_Replicate];
FluxY_ReplicateSummary=[FluxY_ReplicateSummary;FluxY_Replicate];

end

%% plot differentiation rates
figure()
tiledlayout(2,2)
for i = 1:4
    nexttile;
    bar([1, 2, 3], mean_DiffRate(:, i+1));
    hold on
    er = errorbar([1, 2, 3],mean_DiffRate(:, i+1),zeros(size(std_DiffRate(:, i+1))),std_DiffRate(:, i+1));    
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    xticklabels(samples)
    hold off
    title(Fractions(i+1))
end

%% Save the fitting results to csv files
ExcelName= [filefolder,'/Scenario',num2str(GRType),'_',conditions,'_results.xlsx'];
writematrix([["sample", Fractions]; [SampleNames, DifferentialRateSummary]], ExcelName, 'Sheet', 'Differentiation_Rates');
writematrix([["sample", Fractions]; [SampleNames, ResidenceTimeSummary]], ExcelName, 'Sheet', 'Residence_Times');
writematrix([["sample", Fractions]; [SampleNames, TheoryCellNumberSummary]], ExcelName, 'Sheet', 'Simulated_Cell_Numbers');
writematrix([["sample", ["CLP", Fractions]]; [SampleNames, zeros(length(SampleNames),1),FluxX_ReplicateSummary]], ExcelName, 'Sheet', 'Cumulative_Characteristic_Times');
writematrix([["sample", ["CLP", Fractions]]; [SampleNames, ones(length(SampleNames),1), 1+FluxY_ReplicateSummary]], ExcelName, 'Sheet', 'Relative_cell_fluxes');