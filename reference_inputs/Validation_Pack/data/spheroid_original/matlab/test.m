close all
clearvars -except caseid basedir 

if ~exist('caseid'), printHelp; error('caseid must be defined'); end

if ~exist('basedir'),
   basedir='../'; 
end

if strcmp(caseid,'A11-M100')

   cmap=[0,0,0;0,0,1;0,0.5,0;1,0,0];
   load([ basedir 'A11-M100/cross_profiles_along_xpHzperp_chi11_G100.mat']);
   load([ basedir 'A11-M100/cross_profiles_along_xpperp_chi11_G100.mat']);
   load([ basedir 'A11-M100/upar_along_xpa_chi11_G100.mat']);
   load([ basedir 'A11-M100/cp_on_ellipse_chi11_Ga100.mat']);
   
   fig1=figure('Visible','off'); ax1=axes('Parent',fig1); hold(ax1,'on');
   fig2=figure('Visible','off'); ax2=axes('Parent',fig2); hold(ax2,'on');
   fig3=figure('Visible','off'); ax3=axes('Parent',fig3); hold(ax3,'on');
   fig4=figure('Visible','off'); ax4=axes('Parent',fig4); hold(ax4,'on');
 
   for i1 = 1:length(downstream_distance_vec)
      props={'Color',cmap(i1,:),'DisplayName',num2str(downstream_distance_vec(i1))};
      plot(ax1,xpHzperp,uHzperp_along_xpHzperp(:,i1),props{:})
      plot(ax2,xpperp  ,upar_along_xpHzperp(:,i1)   ,props{:})
   end
   plot(ax3,xppar  ,upar_along_xppar,'k')
   set(ax3,'XDir','reverse')

   plot(ax4,th,cp,'k')

   xl=xlabel(ax1,'$x/d$')
   yl=ylabel(ax1,'$u_{rad}$')
   tit=title(ax1,'Profiles of $u_{rad}$ along $x$ ( see figures 7c and 17 b,d)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax1,'on')
   legend(ax1,'toggle');
   figure(fig1);

   xl=xlabel(ax2,'$x/d$')
   yl=ylabel(ax2,'$u_{r||}$')
   tit=title(ax2,'Profiles of $u_{||}$ along $x$ (see figures 7c and 17a,c)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   ll=legend(ax2,'toggle');
   box(ax2,'on')
   set(ll,'Location','SouthEast');
   figure(fig2);

   xl=xlabel(ax3,'$x_{||}$')
   yl=ylabel(ax3,'$u_{||}$')
   tit=title(ax3,'Profiles of $u_{||}$ along $x_{||}$ (see figures 7c and 16a,b)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax3,'on')
   figure(fig3);

   xl=xlabel(ax4,'$\theta$')
   yl=ylabel(ax4,'$C_p$')
   tit=title(ax4,'Pressure distribution on ellipse (see figures 3b and 18)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   set(ax4,'XTick',pi*[-1,-0.5,0,0.5,1],'XtickLabel',{'-\pi','-\pi/2','0','\pi/2','\pi'});
   box(ax4,'on')
   figure(fig4);

elseif strcmp(caseid,'B11-M100')

   cmap=[0,0,0;0,0,1;0,0.5,0;1,0,0];
   load([ basedir 'B11-M100/cross_profiles_along_xpHzperp_chi11_G115.mat']);
   load([ basedir 'B11-M100/cross_profiles_along_xpperp_chi11_G115.mat']);
   load([ basedir 'B11-M100/upar_along_xpa_chi11_G115.mat']);
   
   fig1=figure('Visible','off'); ax1=axes('Parent',fig1); hold(ax1,'on');
   fig2=figure('Visible','off'); ax2=axes('Parent',fig2); hold(ax2,'on');
   fig3=figure('Visible','off'); ax3=axes('Parent',fig3); hold(ax3,'on');
 
   for i1 = 1:length(downstream_distance_vec)
      props={'Color',cmap(i1,:),'DisplayName',num2str(downstream_distance_vec(i1))};
      plot(ax1,xpHzperp,uHzperp_along_xpHzperp(:,i1),props{:})
      plot(ax2,xpperp  ,upar_along_xpperp(:,i1)   ,props{:})
   end
   plot(ax3,xppar  ,upar_along_xppar,'k')
   set(ax3,'XDir','reverse')

   xl=xlabel(ax1,'$x_{pHz\perp}$')
   yl=ylabel(ax1,'$u_{Hz\perp}$')
   tit=title(ax1,'Profiles of $u_{rHz\perp}$ along $x_{pHz\perp}$ (see figure 8f)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   legend(ax1,'toggle');
   box(ax1,'on')
   figure(fig1);

   xl=xlabel(ax2,'$x_{p\perp}$')
   yl=ylabel(ax2,'$u_{r||}$')
   tit=title(ax2,'Profiles of $u_{r||}$ along $x_{\perp}$ (see figure 8c)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   ll=legend(ax2,'toggle');
   set(ll,'Location','SouthEast');
   box(ax2,'on')
   figure(fig2);

   xl=xlabel(ax3,'$x_{p||}$')
   yl=ylabel(ax3,'$u_{r||}$')
   tit=title(ax3,'Profiles of $u_{r||}$ along $x_{p||}$ (see figures 8c,f)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax3,'on')
   figure(fig3);


elseif strcmp(caseid,'B15-M075')

   cmap=[0,0,0;0,0,1;0,0.5,0;1,0,0];
   load([ basedir 'B15-M075/cross_profiles_along_xpHzperp_chi15_G110.mat']);
   load([ basedir 'B15-M075/cross_profiles_along_xpperp_chi15_G110.mat']);
   load([ basedir 'B15-M075/upar_along_xpa_chi15_G110.mat']);
   
   fig1=figure('Visible','off'); ax1=axes('Parent',fig1); hold(ax1,'on');
   fig2=figure('Visible','off'); ax2=axes('Parent',fig2); hold(ax2,'on');
   fig3=figure('Visible','off'); ax3=axes('Parent',fig3); hold(ax3,'on');
 
   for i1 = 1:length(downstream_distance_vec)
      props={'Color',cmap(i1,:),'DisplayName',num2str(downstream_distance_vec(i1))};
      plot(ax1,xpHzperp,uHzperp_along_xpHzperp(:,i1),props{:})
      plot(ax2,xpperp  ,upar_along_xpperp(:,i1)   ,props{:})
   end
   plot(ax3,xppar  ,upar_along_xppar,'k')
   set(ax3,'XDir','reverse')

   xl=xlabel(ax1,'$x_{pHz\perp}$')
   yl=ylabel(ax1,'$u_{Hz\perp}$')
   tit=title(ax1,'Profiles of $u_{rHz\perp}$ along $x_{pHz\perp}$ (see figures 8l)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   legend(ax1,'toggle');
   box(ax1,'on')
   figure(fig1);

   xl=xlabel(ax2,'$x_{p\perp}$')
   yl=ylabel(ax2,'$u_{r||}$')
   tit=title(ax2,'Profiles of $u_{r||}$ along $x_{\perp}$ (see figure 8i)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   ll=legend(ax2,'toggle');
   set(ll,'Location','SouthEast');
   box(ax2,'on')
   figure(fig2);

   xl=xlabel(ax3,'$x_{p||}$')
   yl=ylabel(ax3,'$u_{r||}$')
   tit=title(ax3,'Profiles of $u_{r||}$ along $x_{p||}$ (see figures 8i,l and 21a,b)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax3,'on')
   figure(fig3);

elseif strcmp(caseid,'C15-M075')

   load([ basedir 'C15-M075/partkin_time_evol_chi15_Ga150.mat']);
   
   fig1=figure('Visible','off'); ax1=axes('Parent',fig1); hold(ax1,'on');
   fig2=figure('Visible','off'); ax2=axes('Parent',fig2); hold(ax2,'on');
 
   plot(ax1,upH,upV)
   plot(ax2,t,upH)

   xl=xlabel(ax1,'$u_{pH}$')
   yl=ylabel(ax1,'$u_{pV}$')
   tit=title(ax1,'Phase-space trajectory (see figure 10b)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax1,'on')
   axis(ax1,'equal')
   figure(fig1);

   xl=xlabel(ax2,'$t\,U_g/d$')
   yl=ylabel(ax2,'$u_{pH}$')
   tit=title(ax2,'Time history of $u_{pH}$ (see figure 22)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax2,'on')
   figure(fig2);

elseif strcmp(caseid,'D11-M100')

   load([ basedir 'D11-M100/pdf_particle_vel_chi11_Ga200.mat']);
   load([ basedir 'D11-M100/autocorr_particle_vel_chi11_Ga200.mat']);
   
   fig1=figure('Visible','off'); ax1=axes('Parent',fig1); hold(ax1,'on');
   fig2=figure('Visible','off'); ax2=axes('Parent',fig2); hold(ax2,'on');
   fig3=figure('Visible','off'); ax3=axes('Parent',fig3); hold(ax3,'on');
   fig4=figure('Visible','off'); ax4=axes('Parent',fig4); hold(ax4,'on');
 
   plot(ax1,tau,corr_uv,'k')
   plot(ax1,tau,corr_uh,'r')
   plot(ax2,tau,corr_omv,'k')
   plot(ax2,tau,corr_omh,'r')

   xl=xlabel(ax1,'$\tau\,U_g/d$')
   yl=ylabel(ax1,'$R_{\phi\phi}$')
   tit=title(ax1,'Auto-correlation functions (see figure 13a and 24a)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax1,'on')
   ll=legend(ax1,'$R_{w_p,w_p}$','$R_{u_p,u_p}$')
   set(ll,'Interpreter','Latex')
   figure(fig1);
   xlim(ax1,[0,120])
   ylim(ax1,[-1,1])

   xl=xlabel(ax2,'$\tau\,U_g/d$')
   yl=ylabel(ax2,'$R_{\phi\phi}$')
   tit=title(ax2,'Auto-correlation functions (see figure 13b and 24b)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   ll=legend(ax2,'$R_{\omega_{pz},\omega_{pz}}$','$R_{\omega_{px},\omega_{px}}$')
   set(ll,'Interpreter','Latex')
   box(ax2,'on')
   xlim(ax2,[0,120])
   ylim(ax2,[-1,1])
   figure(fig2);

elseif strcmp(caseid,'D15-M500')

   load([ basedir 'D15-M500/pdf_particle_vel_chi15_Ga220.mat']);
   load([ basedir 'D15-M500/autocorr_particle_vel_chi15_Ga220.mat']);
   
   fig1=figure('Visible','off'); ax1=axes('Parent',fig1); hold(ax1,'on');
   fig2=figure('Visible','off'); ax2=axes('Parent',fig2); hold(ax2,'on');
   fig3=figure('Visible','off'); ax3=axes('Parent',fig3); hold(ax3,'on');
   fig4=figure('Visible','off'); ax4=axes('Parent',fig4); hold(ax4,'on');
 
   plot(ax1,tau,corr_uv,'k')
   plot(ax1,tau,corr_uh,'r')
   plot(ax2,tau,corr_omv,'k')
   plot(ax2,tau,corr_omh,'r')

   xl=xlabel(ax1,'$\tau\,U_g/d$')
   yl=ylabel(ax1,'$R_{\phi\phi}$')
   tit=title(ax1,'Auto-correlation functions (see figure 13c and 24c)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   box(ax1,'on')
   ll=legend(ax1,'$R_{w_p,w_p}$','$R_{u_p,u_p}$')
   set(ll,'Interpreter','Latex')
   figure(fig1);
   xlim(ax1,[0,120])
   ylim(ax1,[-1,1])

   xl=xlabel(ax2,'$\tau\,U_g/d$')
   yl=ylabel(ax2,'$R_{\phi\phi}$')
   tit=title(ax2,'Auto-correlation functions (see figure 13d and 24d)')
   set(tit,'Interpreter','Latex')
   set(xl,'Interpreter','Latex')
   set(yl,'Interpreter','Latex')
   ll=legend(ax2,'$R_{\omega_{pz},\omega_{pz}}$','$R_{\omega_{px},\omega_{px}}$')
   set(ll,'Interpreter','Latex')
   box(ax2,'on')
   xlim(ax2,[0,120])
   ylim(ax2,[-1,1])
   figure(fig2);

else
   error('caseid wrong value')
end
%-------------------------------------------------------------------------------
function printHelp()
disp('Set caseid to perform test on:')
disp(' caseid=''A11-M100'': Steady vertical regime chi=1.1, Ga=100, mstar=1 (rhop/rhof=2.1)')
disp(' caseid=''B11-M100'': Steady oblique regime chi=1.1, Ga=115, mstar=1 (rhop/rhof=2.1)')
disp(' caseid=''B15-M075'': Steady oblique regime chi=1.5, Ga=110, mstar=0.75 (rhop/rhof=2.14)')
disp(' caseid=''C15-M075'': Vertical periodic regime chi=1.5, Ga=150, mstar=0.75 (rhop/rhof=2.14)')
disp(' caseid=''D11-M100'': Chaotic regime chi=1.1, Ga=200, mstar=1 (rhop/rhof=2.1)')
disp(' caseid=''D15-M500'': Chaotic regime chi=1.5, Ga=220, mstar=5 (rhop/rhof=14.32)')
return
end

