close all;clear all;
set(0,'DefaultAxesFontName','Helvetica','DefaultAxesFontSize',10);
set(0,'defaultaxesbox','on');
set(0,'DefaultAxesTickDir','out');
set(0,'defaultlinelinewidth',1);
%%-----------------------------------------------------------------%
%% matlab script for reading the data provided in this repository.
%%
%% 
%% [1] M. Uhlmann and J. Dusek, "The motion of a single heavy sphere in
%% ambient fluid: a benchmark for interface-resolved particulate flow
%% simulations with significant relative velocities", Int. J. Multiphase
%% Flow, 59:221-243, 2014. 
%% http://dx.doi.org/10.1016/j.ijmultiphaseflow.2013.10.010
%% http://arxiv.org/abs/1310.5358
%%
%% markus.uhlmann@kit.edu
%%-----------------------------------------------------------------%
%% 09/10/2013 first version posted
%% 11/04/2014 updates script
%%-----------------------------------------------------------------%
iplot=1; % '1'=>all data will be plotted 
%
% choice of flow case: [choose one out of A-D]
%case_flag='A'; %<-G=144
case_flag='B'; %<-G=178.46
%case_flag='C'; %<-G=190
%case_flag='D'; %<-G=250
%
% adapt the following setting to the location where the data is stored 
% (top-level directory):
basedir='/home/USERNAME/tmp/';
%
if strcmp(case_flag,'A')
  idir=[basedir 'G144/']
  % read relative velocity deficit on axis:
  ifile='relvel_deficit_axis_G144.mat'
  load([idir ifile]);
  % read cross profiles at various downstream distances:
  ifile='cross_profiles_along_xpperp_G144.mat';
  load([idir ifile]);
  ifile='cross_profiles_along_xpHzperp_G144.mat';
  load([idir ifile]);
  % read cp on great circles:
  ifile='cp_on_great_circles_G144.mat';
  load([idir ifile]);
elseif strcmp(case_flag,'B')
  idir=[basedir 'G178/']
  % read relative velocity deficit on axis:
  ifile='relvel_deficit_axis_G178.mat'
  load([idir ifile]);
  % read cross profiles at various downstream distances:
  ifile='cross_profiles_along_xpperp_G178.mat';
  load([idir ifile]);
  ifile='cross_profiles_along_xpHzperp_G178.mat';
  load([idir ifile]);
  % read cp on great circles:
  ifile='cp_on_great_circles_G178.mat';
  load([idir ifile]);
elseif strcmp(case_flag,'C')
  idir=[basedir 'G190/']
  % read time evolution of particle velocity data:
  ifile='partvel_time_evol_G190.mat';
  load([idir ifile]);
  % read data for one out of 5 snapshots over one oscillation period:  
  nsnap=3; %range: 1-5
  % read cp on great circles:
  ifile=sprintf('cp_on_great_circles_G190_t%d.mat',nsnap);
  load([idir ifile]);
elseif strcmp(case_flag,'D')
  idir=[basedir 'G250/']
  % read particle velocity statistics: pdfs & auto-correlations
  ifile='pdf_particle_vel_G250.mat';
  load([idir ifile]);
  ifile='autocorr_particle_vel_G250.mat';
  load([idir ifile]);
end
%
title_string=sprintf('Uhlmann & Dusek (IJMF 2014), case %s, G=%g, D_d_o_m_a_i_n=%g',...
case_flag,galileo,domain_diameter_D_Omega);
%-------------------------------------------------------------------------
% PLOTS
%-------------------------------------------------------------------------
if iplot==1
  if ~strcmp(case_flag,'D')
    if strcmp(case_flag,'C')
      % find the index corresponding to the time when the selected
      % snapshot "nsnap" is taken: 
      ind_snap=find(abs(time_evol-time_snapshot)==min(abs(time_evol-time_snapshot)));
      %-------------------------------------------------------------------------
      % time evolution of the vertical particle velocity component
      f1a=figure;a1a=axes;hold on;
      %
      plot(time_evol,uv,'k-');
      % the symbol marks the time when the snapshot is taken:
      plot(time_evol(ind_snap),uv(ind_snap),'ro','LineWidth',3);
      %
      title(title_string);
      xlabel('time');ylabel('u_p_V');
      set(a1a,'XLim',[0 50],'YLim',[-1.387 -1.365]);
      %-------------------------------------------------------------------------
      % time evolution of the horizontal particle velocity component
      f1b=figure;a1b=axes;hold on;
      %
      plot(time_evol,uh,'k-');
      % the symbol marks the time when the snapshot is taken:
      plot(time_evol(ind_snap),uh(ind_snap),'ro','LineWidth',3);
      %
      title(title_string);
      xlabel('time');ylabel('u_p_H');
      set(a1b,'XLim',[0 50],'YLim',[0.093 0.18]);
      %-------------------------------------------------------------------------
      % time evolution of the horizontal *angular* particle velocity component
      f1c=figure;a1c=axes;hold on;
      %
      plot(time_evol,omh,'k-');
      % the symbol marks the time when the snapshot is taken:
      plot(time_evol(ind_snap),omh(ind_snap),'ro','LineWidth',3);
      %
      title(title_string);
      xlabel('time');ylabel('\omega_p_H');
      set(a1c,'XLim',[0 50],'YLim',[0.003 0.0213]);
      %-------------------------------------------------------------------------
    end %only case "C"
    if ~strcmp(case_flag,'C')
      %-------------------------------------------------------------------------
      % relative velocity deficit along the axis
      f2=figure;a2=axes;hold on;
      plot(x_ppar_upstream,u_rpar_upstream,'k-');
      plot(x_ppar_downstream,u_rpar_downstream,'k-');
      plot([-20 100],[0 0],'r:');
      title(title_string);
      xlabel('x_p_,_p_a_r');ylabel('u_r_,_p_a_r/u_r_,_p_a_r_,_\infty');
      set(a2,'XLim',[-2 10],'YLim',[-.4 1.1]);
      %-------------------------------------------------------------------------
      %-------------------------------------------------------------------------
      % relative velocity deficit along the axis - definition as in 
      % garcia-villalba,kidanemariam,uhlmann (ijmf 2011)
      % http://dx.doi.org/10.1016/j.ijmultiphaseflow.2012.05.007
      % equation (9)
      % NOTE: x-axis scaled with radius, for consistency with above paper
      % 
      f2b=figure;a2b=axes;hold on;
      %
      ud0=(1-u_rpar_downstream);
      plot(x_ppar_downstream*2,ud0,'k-');
      %
      % to guide the eye: line with slope x^-{2/3} (turbulent wake)
      xx=1+30*(0:100)/100;
      plot(xx,2.5*xx.^(-2/3),'r--');
      %
      plot([-20 100],[0 0],'r:');
      title(title_string);
      xlabel('2x_p_,_p_a_r');ylabel('1-u_r_,_p_a_r/u_r_,_p_a_r_,_\infty');
      set(a2b,'XScale','log','YScale','log');
      set(a2b,'XLim',[1 1.e2],'YLim',[1.e-2 2.e0]);
      legend('spectral-element DNS','x^-^2^/^3');
      %-------------------------------------------------------------------------
      % *CROSS-PROFILES OF VELOCITY*
      %--------------------------------
      % A) data along the direction 'epperp', 
      %    at: 'xphzperp'=0 and various values of downstream distance ('xppar')
      %
      %% one figure per quantity, overlay curves at 
      %% different downstream distances
      xtick_vec=[-3 -2 -1 0 1 2 3];
      profile_xlim=[-2.5 +2.5];
      %%----------------
      upar_ylim=[-.5 1.75];
      upar_ytick_vec=[-.4 -.2 0 .2 .4 .6 .8 1. 1.2 1.4 1.6];
      uHzperp_ylim=[-.1 .1];
      uHzperp_ytick_vec=[-.1 -.08 -.06 -.04 -.02 0 .02 .04 .06 .08 .1];
      uperp_ylim=uHzperp_ylim;
      uperp_ytick_vec=uHzperp_ytick_vec;
      %
      profile_color_vec=['k' 'b' 'g' 'r' 'c' 'm'];
      %%------------------------------------------------------------------
      f3a=figure;a3a=axes;hold on;
      for jj=1:length(downstream_distance_vec)
        lines=[profile_color_vec(jj) '-'];
        plot(xpperp,squeeze(upar_along_xpperp(:,jj)),lines);
      end
      legend(sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(1)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(2)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(3)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(4)));
      title(title_string);
      xlabel('x_p_,_p_e_r_p');ylabel('u_r_,_p_a_r');
      set(a3a,'YLim',upar_ylim,'XLim',profile_xlim);
      set(a3a,'XTick',xtick_vec,'YTick',upar_ytick_vec);
      %%------------------------------------------------------------------
      f3b=figure;a3b=axes;hold on;
      for jj=1:length(downstream_distance_vec)
        lines=[profile_color_vec(jj) '-'];
        plot(xpperp,squeeze(uHzperp_along_xpperp(:,jj)),lines);
      end
      legend(sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(1)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(2)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(3)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(4)));
      title(title_string);
      xlabel('x_p_,_p_e_r_p');ylabel('u_r_,_H_z_,_p_e_r_p');
      set(a3b,'YLim',uHzperp_ylim,'XLim',profile_xlim);
      set(a3b,'XTick',xtick_vec,'YTick',uHzperp_ytick_vec);
      %%------------------------------------------------------------------
      f3c=figure;a3c=axes;hold on;
      for jj=1:length(downstream_distance_vec)
        lines=[profile_color_vec(jj) '-'];
        plot(xpperp,squeeze(uperp_along_xpperp(:,jj)),lines);
      end
      legend(sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(1)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(2)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(3)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(4)));
      title(title_string);
      xlabel('x_p_,_p_e_r_p');ylabel('u_r_,_p_e_r_p');
      set(a3c,'YLim',uperp_ylim,'XLim',profile_xlim);
      set(a3c,'XTick',xtick_vec,'YTick',uperp_ytick_vec);
      %%------------------------------------------------------------------
      % B) data along the direction 'ephzperp', 
      %    at: 'xpperp'=0 and various values of downstream distance ('xppar')
      %%------------------------------------------------------------------
      f4a=figure;a4a=axes;hold on;
      for jj=1:length(downstream_distance_vec)
        lines=[profile_color_vec(jj) '-'];
        plot(xpHzperp,squeeze(upar_along_xpHzperp(:,jj)),lines);
      end
      legend(sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(1)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(2)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(3)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(4)));
      title(title_string);
      xlabel('x_p_,_H_z_,_p_e_r_p');ylabel('u_r_,_p_a_r');
      set(a4a,'YLim',upar_ylim,'XLim',profile_xlim);
      set(a4a,'XTick',xtick_vec,'YTick',upar_ytick_vec);
      %%------------------------------------------------------------------
      f4b=figure;a4b=axes;hold on;
      for jj=1:length(downstream_distance_vec)
        lines=[profile_color_vec(jj) '-'];
        plot(xpHzperp,squeeze(uHzperp_along_xpHzperp(:,jj)),lines);
      end
      legend(sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(1)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(2)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(3)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(4)));
      title(title_string);
      xlabel('x_p_,_H_z_,_p_e_r_p');ylabel('u_r_,_H_z_,_p_e_r_p');
      set(a4b,'YLim',uHzperp_ylim,'XLim',profile_xlim);
      set(a4b,'XTick',xtick_vec,'YTick',uHzperp_ytick_vec);
      %%------------------------------------------------------------------
      f4c=figure;a4c=axes;hold on;
      for jj=1:length(downstream_distance_vec)
        lines=[profile_color_vec(jj) '-'];
        plot(xpHzperp,squeeze(uperp_along_xpHzperp(:,jj)),lines);
      end
      legend(sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(1)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(2)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(3)),...
             sprintf('x_p_,_p_a_r=%g',downstream_distance_vec(4)));
      title(title_string);
      xlabel('x_p_,_H_z_,_p_e_r_p');ylabel('u_r_,_p_e_r_p');
      set(a4c,'YLim',uperp_ylim,'XLim',profile_xlim);
      set(a4c,'XTick',xtick_vec,'YTick',uperp_ytick_vec);
      %%------------------------------------------------------------------
    end % not case "C"
    %-------------------------------------------------------------------------
    %% pressure profiles on 'great circles':
    % note: for axisymmetric, curve is identical -> plot only one of them
    f1=figure;a1=axes;hold on;
    %
    plot(theta1,cp1,'b-');
    plot(theta2,cp2,'k-');
    %
    legend('\theta_1','\theta_2');
      title(title_string);
    xlabel('\theta');ylabel('c_p');
    set(a1,'TickDir','out','Box','on'); 
    set(a1,'XLim',[-1.05 1.05]*pi,'YLim',[-1 1.5]);
    set(a1,'XTick',[-pi -pi/2 0 pi/2 pi]);
    set(a1,'XTickLabel',{'-p' '-p/2' '0' 'p/2' 'p'},'FontName','symbol');
    %-------------------------------------------------------------------------
  else % case "D"
    %---------------------------------------------------------
    % STAT --- PLOTS 
    %---------------------------------------------------------
    pdf_xlim=[-4 +4];
    pdf_xtick=[-4 -2 0 2 4];
    pdf_ylim=[1e-3 1];
    pdf_ytick=[1e-5 1e-4 1e-3 1e-2 1e-1 1];
    %---------------------------------------------------------
    %% pdf of linear particle velocity - vert. & horiz. components:
    fpdf(1)=figure;apdf(1)=axes;hold on;
    %
    plot(bins_uv,pdf_uv,'k-');
    plot(bins_uh,pdf_uh,'r-');
    %
    %the plot a gaussian:
    na=500;xa=-5+10*(0:na)/na;
    plot(xa,1/(1*sqrt(2*pi))*exp(-xa.^2/(2*1^2)),'b--');
    %
    legend('u_p_V','u_p_H');
    title(title_string);
    xlabel('u"/\sigma');ylabel('\sigma pdf');
    set(apdf(1),'YScale','log','XScale','lin');
    set(apdf(1),'XLim',pdf_xlim,'YLim',pdf_ylim);
    set(apdf(1),'XTick',pdf_xtick,'YTick',pdf_ytick);
    %---------------------------------------------------------
    %% pdf of *ANGULAR* particle velocity - vert. & horiz. components:
    fpdf(2)=figure;apdf(2)=axes;hold on;
    %
    plot(bins_omv,pdf_omv,'k-');
    plot(bins_omh,pdf_omh,'r-');
    %
    %the plot a gaussian:
    na=500;xa=-5+10*(0:na)/na;
    plot(xa,1/(1*sqrt(2*pi))*exp(-xa.^2/(2*1^2)),'b--');
    %
    legend('\omega_p_V','\omega_p_H');
    title(title_string);
    xlabel('\omega"/\sigma');ylabel('\sigma pdf');
    set(apdf(2),'YScale','log','XScale','lin');
    set(apdf(2),'XLim',pdf_xlim,'YLim',pdf_ylim);
    set(apdf(2),'XTick',pdf_xtick,'YTick',pdf_ytick);
    %---------------------------------------------------------
    % CORR --- PLOTS 
    %---------------------------------------------------------
    corr_xlim=[0 120];
    corr_xtick=(0:50:200);
    corr_ylim=[-.4 1];
    corr_ytick=(-.4:.2:1);
    %---------------------------------------------------------
    %% auto-correlations of linear particle velocity - vert. & horiz. components:
    fcorr(1)=figure;acorr(1)=axes;hold on;
    %
    plot(inc_uv,corr_uv,'k-');
    plot(inc_uh,corr_uh,'r-');
    %
    legend('u_p_V','u_p_H');
    title(title_string);
    xlabel('\tau_s_e_p');ylabel('R_u_u');
    set(acorr(1),'XLim',corr_xlim,'YLim',corr_ylim);
    set(acorr(1),'XTick',corr_xtick,'YTick',corr_ytick);
    %---------------------------------------------------------
    %% auto-correlations of *angular* particle velocity - vert. & horiz. components:
    fcorr(2)=figure;acorr(2)=axes;hold on;
    %
    plot(inc_omv,corr_omv,'k-');
    plot(inc_omh,corr_omh,'r-');
    %
    legend('\omega_p_V','\omega_p_H');
    title(title_string);
    xlabel('\tau_s_e_p');ylabel('R_\omega_\omega');
    set(acorr(2),'XLim',corr_xlim,'YLim',corr_ylim);
    set(acorr(2),'XTick',corr_xtick,'YTick',corr_ytick);
    %---------------------------------------------------------
    %---------------------------------------------------------
  end
end