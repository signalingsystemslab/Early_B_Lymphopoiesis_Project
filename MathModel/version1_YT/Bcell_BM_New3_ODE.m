%(c) 2020 Signaling Systems Lab UCLA
%All rights reserved. 
%This MATLAB code implements the mathematical modeling on the B-cell population
%dynamics. It fitted the birth-death ODE to the cell number measurements
%and inferred the differential rates.

%A detailed application of this package is given in the main text. 

%% Specify the cell population names
clear;
filefolder=pwd;
Name={'Fr.A','Fr.B','Fr.C','Fr.C''','Fr.D','Fr.E'};
SpecieNum=size(Name,2);

%% Load data: cell number measurements, and proliferation, death rates
color=jet(SpecieNum);

[WT, txt]= xlsread('Total01102019v1.xlsx',1,'Y3:AD18');
[KO, txt]= xlsread('Total01102019v1.xlsx',1,'Y20:AD33');
[KO2, txt]= xlsread('Total01102019v1.xlsx',1,'Y35:AD42');



newdata=1;
if newdata==1
[GR_WT, txt]= xlsread('ParameterSummaryV2_5.xlsx',2,'B2:G2');
[GR_KO, txt]= xlsread('ParameterSummaryV2_5.xlsx',2,'B3:G3');
GR_WT=GR_WT*100;GR_KO=GR_KO*100;
[DR_WT, txt]= xlsread('ParameterSummaryV2_5.xlsx',1,'B2:G3');
[DR_KO, txt]= xlsread('ParameterSummaryV2_5.xlsx',2,'B3:G3');
DR_WT=DR_WT*0;DR_KO=DR_KO*0;
end

GRTypeTotal=3;

DRSummary=cell(3,GRTypeTotal);
GRSummary=cell(3,GRTypeTotal);
FluxNew_X=cell(3,GRTypeTotal);
FluxNew_X_Final=cell(3,GRTypeTotal);
FluxNew_Y=cell(3,GRTypeTotal);
Frac1_Summary=cell(3,GRTypeTotal);
Frac2_Summary=cell(3,GRTypeTotal);

%% Input the proliferation and death rates
DifferentialRateSummary=[];TheoryCellNumberSummary=[];TheoryFluxSummary=cell(1,3);
FluxX_ReplicateSummary=[];FluxY_ReplicateSummary=[];
ResidenceTimeSummary=[];
for GRType=1:GRTypeTotal% The type of net-proliferation rates used
%Input growth rate indicated by experiment...

% h1=figure(1);clf;set(h1,'position', [200, 200, 800, 600])
 
for DataType=1:3 %For the 3 genotypes
if DataType==1
    Data=WT; figurename=[filefolder,'\WT'];
    ManualGR=[10 33 24 13 75.1 35 3 0]/100;
    if GRType==1
    GRToUse=mean(GR_WT,1)/100;
    else
    GRToUse=mean(GR_WT-DR_WT,1)/100;
    end
elseif  DataType==2
    Data=KO; figurename=[filefolder,'\KO'];
    ManualGR=[10 5 6 13 25 5 0.3 0]/100;
    if GRType==1
    GRToUse=mean(GR_KO,1)/100;
    else
    GRToUse=mean(GR_KO-DR_KO,1)/100;%GRToUse=mean(GR_WT-DR_WT,1)/100;
    end     
elseif  DataType==3
    Data=KO2; figurename=[filefolder,'\KO2'];
    ManualGR=[10 33 24 13 75.1 35 3 0]/100;%assume it's the same as WT first; [10 5 6 13 25 5 0.3 0]/100;
    if GRType==1
    GRToUse=mean(GR_KO,1)/100;
    else
    GRToUse=mean(GR_KO,1)/100;%mean(GR_KO-DR_KO,1)/100;%GRToUse=mean(GR_KO-DR_KO,1)/100;%GRToUse=mean(GR_WT-DR_WT,1)/100;
    end    
end

Data(:,isnan(Data(1,:))) = [];
%A(isnan(A), 2),:)=[];
Conversion_WT=[1 1e4 1e4 1e4 1e4 1e5 1e5];% 1e5];
Conversion_WT(:,1)=[]; %Delete CLP
Data=Data.*Conversion_WT;




Mean_WT=mean(Data,1);
Variance_WT=var(Data,1);
Sd_WT=std(Data,1);
Xaxis=1:1:size(Data,2);
X_WT=Mean_WT+Sd_WT;
%X(isnan(X)) = [];
yLimHigh_WT=9*1e6;%max(X_WT);

%% Specify the type of net-proliferation rates used for model fitting
 if GRType<3
    GR=GRToUse;
 elseif GRType==3
    GR=ones(1,size(Mean_WT,2));%*0.5;%mean(GRToUse)*ones(1,size(Mean_WT,2));
 elseif  GRType==4
     GR=mean(GRToUse)*ones(1,size(Mean_WT,2))/2;
 elseif  GRType==5
     GR=mean(GRToUse)*ones(1,size(Mean_WT,2))*2;
  
 end

%% Calculate the inferred the differential rates based on the model: the anlytical formula for the differential rates are used (see main text for detail).

%GR_WT=[0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25];
%DR_WT=[0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75];
%Differentiation rate by fitted formula
DifferentiaRate=1;%Input the first one
Frac1=1;
Frac2=0;
Theory1_WT=Mean_WT(1);
TheoryFlux=1000;
ResideTime=1/(DifferentiaRate-GR(1));
SumTemp=0;
for i=2:size(GR,2) 
    DifferentiaRate=[DifferentiaRate DifferentiaRate(end)*Mean_WT(i-1)/Mean_WT(i)+GR(i)];
    Frac1=[Frac1,Mean_WT(1)/Mean_WT(i)];
    Frac2=[Frac2,sum(GR(2:i).*Mean_WT(2:i))/Mean_WT(i)];
    if DifferentiaRate(i)-GR(i)~=0
        Theory1_WT=[Theory1_WT,DifferentiaRate(i-1)/(DifferentiaRate(i)-GR(i))*Theory1_WT(end)];
        TheoryFlux=[TheoryFlux,DifferentiaRate(i-1)/(DifferentiaRate(i)-GR(i))*TheoryFlux(end)];
        ResideTime=[ResideTime,1/(DifferentiaRate(i)-GR(i))];
    else
        display('singularity');
    end
  
end
GRSummary{DataType,GRType}=GR;
DRSummary{DataType,GRType}=DifferentiaRate;
FluxNew_X{DataType,GRType}=[1/(DifferentiaRate(1)-GR(1)),1./(DifferentiaRate-GR)];%,1./DifferentiaRate];
FluxNew_Y{DataType,GRType}=[-Theory1_WT(1)*(GR(1)-DifferentiaRate(1)),DifferentiaRate.*Theory1_WT];

ResideTimeSummary{DataType,GRType}=ResideTime;
Frac1_Summary{DataType,GRType}=Frac1;
Frac2_Summary{DataType,GRType}=Frac2;


%% Fit differential rates for each replicate and export data

if GRType==1  % 1 or 3(change average GR) ... change the data save for different GRType
Frac1=1;
Frac2=0;
TheoryCellNumber=Data;
DifferentialRateReplicate=Data;
ResidenceTimeReplicate=Data;
FluxX_Replicate=Data;
FluxY_Replicate=Data;
for kk=1:size(Data,1)
DifferentiaRate=1;%Input the first one
Theory1_WT2=Data(kk,1);
SumTemp=0;
for i=2:size(GR,2) 
    DifferentiaRate=[DifferentiaRate DifferentiaRate(end)*Data(kk,i-1)/Data(kk,i)+GR(i)];
    Frac1=[Frac1,Data(kk,1)/Data(kk,i)];
    Frac2=[Frac2,sum(GR(2:i).*Data(kk,2:i))/Data(kk,i)];
    if DifferentiaRate(i)-GR(i)~=0
        Theory1_WT2=[Theory1_WT2,DifferentiaRate(i-1)/(DifferentiaRate(i)-GR(i))*Theory1_WT2(end)];
    else
        display('singularity');
    end
  
end
TheoryCellNumber(kk,:)=Theory1_WT2;
DifferentialRateReplicate(kk,:)=DifferentiaRate;
ResidenceTimeReplicate(kk,:)=1./(DifferentiaRate-GR);
FluxX_Replicate(kk,:)= cumsum(1./DifferentiaRate);
FluxY_Replicate(kk,:)= cumsum(GR.*Theory1_WT2)/Theory1_WT2(1);
end

TheoryCellNumberSummary=[TheoryCellNumberSummary;zeros(1,size(TheoryCellNumberSummary,2));TheoryCellNumber];

DifferentialRateSummary=[DifferentialRateSummary;zeros(1,size(DifferentialRateSummary,2));DifferentialRateReplicate];

ResidenceTimeSummary=[ResidenceTimeSummary;zeros(1,size(ResidenceTimeSummary,2));ResidenceTimeReplicate];

FluxX_ReplicateSummary=[FluxX_ReplicateSummary;zeros(1,size(FluxX_ReplicateSummary,2));FluxX_Replicate];
FluxY_ReplicateSummary=[FluxY_ReplicateSummary;zeros(1,size(FluxY_ReplicateSummary,2));FluxY_Replicate];

end



%% Parameter sensitivy analysis: perturb WT differentiation rates of B, C, C'
%by decrease 2-fold, increase 10-fold, increase 10-fold, and predict cell number
%fraction C' proliferation 5-fold lower.
PerturbDR_index=[3 4]; %2,3,4
PerturbGR_index=[3 4]; %4
if DataType==1
    
    Perturb_WT=Mean_WT(1);
    %DifferentiaRate=DRSummary{DataType,GRType};
    DifferentiaRate=ones(1,size(DRSummary{DataType,GRType},2))*mean(DRSummary{DataType,GRType});
    
    DifferentiaRate(PerturbDR_index)=DifferentiaRate(PerturbDR_index);%for DR1=0.5:/2, *10, *10,/1.2|| for DR1=0.5:/1.5, *10, *10,/1.2
    display(GR);
    GR=2.7*GR;%in order to make the figure for Eason Panel 2
    %Need to set GRType=1
    %display(DifferentiaRate);
    GR(PerturbGR_index)=GR(PerturbGR_index)/10;
    for i=2:size(GR,2) 
    if DifferentiaRate(i)-GR(i)~=0
        Perturb_WT=[Perturb_WT,DifferentiaRate(i-1)/(DifferentiaRate(i)-GR(i))*Perturb_WT(end)];
    else
        display('singularity');
    end
    end
    
end

%% Plotting the figures 
figure
%bar(Xaxis,Mean_WT);
%subplot(2,2,1)
errorbar(Xaxis,Mean_WT,Sd_WT,'o','markersize',15,'Color',color(1,:),'MarkerFaceColor',color(1,:));hold on;
plot(Xaxis,Theory1_WT,'-','linewidth',2,'color','k');
%plot(Xaxis,Perturb_WT,'-','linewidth',2,'color','k');
 xticks(Xaxis);xticklabels(Name);
 ylabel('Cell count');
ylim([0 yLimHigh_WT]);
xlim([min(Xaxis)-1 max(Xaxis)+1]);
set(gca, 'linewidth', 4);
set(gca,'FontSize',26);

%dd
TheoryFluxSummary{1,DataType}=TheoryFlux;

end
 if GRType==1
 figurename2=['NumberUniformGR.jpg'];
 elseif  GRType==2
 figurename2=['NumberNonUniGR.jpg'];
 end
 saveas(gcf,figurename2); 
end


for i=1:3
    h1=figure(i+1);clf;set(h1,'position', [-1000, 100, 800, 600])
    bar(DRSummary{i,2}); 
     ylabel('Differentiation rate')
    xticks(Xaxis);xticklabels(Name);
    set(gca,'FontSize',26);
     figurename2=['DR_',num2str(i),'.jpg'];
 saveas(gcf,figurename2); 
end


for i=1:3
    h1=figure(i+1);clf;set(h1,'position', [-1000, 100, 800, 600])
    bar(GRSummary{i,2}); 
    xticks(Xaxis);xticklabels(Name);
     ylabel('Net proliferation rate')
    set(gca,'FontSize',26);
     figurename2=['GR_',num2str(i),'.jpg'];
 saveas(gcf,figurename2); 
end



%%
%flux accumulation
 %h1=figure(1);clf;set(h1,'position', [-1000, 100, 800, 600])
 CellFlux=cell(1,3);
 FluxPlox_X=[];%cell(1,3);
 FluxPlox_Y=[];%cell(1,3);
%cc=jet(3);
for ii=1:3
FluxNew_X_Final{ii,2}=cumsum(FluxNew_X{ii,2})-FluxNew_X{ii,2}(1);
%plot( FluxNew_X_Final{ii,2},FluxNew_Y{ii,2},'o-','linewidth',2,'color',cc(ii,:));hold on;
end


%% Save the fitting results to csv files
ExcelName=[filefolder,'/DifferentialRatesReplicate.csv'];
csvwrite(ExcelName,DifferentialRateSummary);
ExcelName=[filefolder,'/ResidenceTimeReplicate.csv'];
csvwrite(ExcelName,ResidenceTimeSummary);
ExcelName=[filefolder,'/SimulatedCellNumber.csv'];
csvwrite(ExcelName,TheoryCellNumberSummary);

ExcelName=[filefolder,'/FluxPlotX_new.csv'];
csvwrite(ExcelName,FluxNew_X_Final);
ExcelName=[filefolder,'/FluxPlotY_new.csv'];
csvwrite(ExcelName,FluxNew_Y);


 close all;
 
  
 
