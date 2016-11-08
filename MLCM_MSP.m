function [tout] = MLCM_MSP(Y,X_betas,zs,Gsim,burnin,thin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PURPOSE: MCMC for two-component multivariate latent class model with weight depending on
%         high-deminsional covaraites and each component following
%         mulitivariate normal distributions.
%INPUT:   Y        n by s responses
%         X_betas  n by p covaraites
%         zs       n by 1 initial classification
%         Gsim     1 by 1 number of simulations
%         burnin   1 by 1 number of burn in 
%         thin     1 by 1 number of thin
%OUTPUT:  betas_0_out  (Gsim-burnin)/thin by p+1 betas
%         thetas_out   (Gsim-burnin)/thin by 2*s thetas
%         Sigmas_out   (Gsim-burnin)/thin by s*(s+1) Sigmas
%                       upper triangular by rows 
%         zs_out       (Gsim-burnin)/thin by n zs
%         tout=[betas_0_out,thetas_out,Sigmas_out,zs_out]
%Writen by: Bin Zhu, based on Maclehose MSP codes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[n,s]=size(Y);
[n,p]=size(X_betas);

X=[ones(n,1),X_betas]; % add dummy covariate for the intercept;


  a0=30;
  b0=a0;
  a1=6.5;
  b1=a1;
  c=0;
  %d=4;      %95% in range [-3.92,3.92]
  d=0.1507;  %99% in range [-1,1]
  c0=0;      %  prior mean for intercept
  cv0=100;    % prior variance for intercept
  a=1;        % alpha value
  
  %construct prior for the latent class mean and covariance matrix;
  rho_1=6;
  rho_2=rho_1; 
  psi_1=1;
  psi_2=psi_1;
  
  thetas_0=mean(Y);
  Sigma_0=(rho_1-s-1)/2.*cov(Y);
  
  thetas_0_1=thetas_0;
  Sigma_0_1=Sigma_0;  
  thetas_0_2=thetas_0; 
  Sigma_0_2=Sigma_0;

  
  % constants and latent vars for logistic approximation 
  nu = 7.3;                       % df
  sigma = pi*sqrt((nu-2)/(3*nu));    % standard deviation
  phis = ones(n,1);                % latent variable in normal-gamma mix
  sds = sigma*phis.^(-0.5);
  gs = zeros(n,1);

  betas_0= zeros(p+1,1);
  %muG = 0;  
  ks=ones(p,1); % Bin index for DP;
  mus=zeros(p,1);
  mus_0=[c0,mus']';
  p_star=max(ks);              %no. grps
  Ssum=(repmat(ks(:,1),[1 p_star])-repmat(1:p_star,[p 1])==0);
  ms = ones(1,p)*Ssum;              %no in each grp.
  
  betas_0_out=zeros((Gsim-burnin)/thin,p+1);
  %mus_out=zeros((Gsim-burnin)/thin,p);
  %taus_out=zeros((Gsim-burnin)/thin,p);
  
  thetas_out=zeros((Gsim-burnin)/thin,2*s);
  Sigmas_out=zeros((Gsim-burnin)/thin,s*(s+1));
  
  zs_out=zeros((Gsim-burnin)/thin,n);
  %phis_out=zeros((Gsim-burnin)/thin,length(zs));
  %lambdas_out=zeros((Gsim-burnin)/thin,p);
  
  %ks_out=zeros((Gsim-burnin)/thin,p);
  
  lambdas=ones(p,1);
  iLambda=inv(diag([cv0,lambdas']));
  XpX=X'*X;
  
  taus_star=1;
  mus_star=0;
  taus=ones(p,1);

for g=1:Gsim
  % --------------------------------------------------------------------- %
  % update thetas_1/2 and Sigma_1/2;
  % -------------------------------------------------------------------- % 
  Y_bar_1=mean(Y(find(zs==1),:));
  n_1=sum(zs==1);
  tmp=Y(find(zs==1),:)- Y_bar_1(ones(1,n_1),:);
  S_1=tmp'*tmp;
  
  tmp=n_1/(n_1+psi_1);
  theta_hat_1=tmp*Y_bar_1+(1-tmp)*thetas_0_1;
  psi_hat_1=psi_1+n_1;
  rho_hat_1=rho_1+n_1;
  Sigma_hat_1=Sigma_0_1+S_1+n_1/(1+n_1/psi_1).*(Y_bar_1-thetas_0)'*(Y_bar_1-thetas_0);
  
  Y_bar_2=mean(Y(find(zs==0),:));
  n_2=sum(zs==0);
  tmp=Y(find(zs==0),:)- Y_bar_2(ones(1,n_2),:);
  S_2=tmp'*tmp;
  
  tmp=n_2/(n_2+psi_2);
  theta_hat_2=tmp*Y_bar_2+(1-tmp)*thetas_0_2;
  psi_hat_2=psi_2+n_2;
  rho_hat_2=rho_2+n_2;
  Sigma_hat_2=Sigma_0_2+S_2+n_2/(1+n_2/psi_2).*(Y_bar_2-thetas_0)'*(Y_bar_2-thetas_0);

  Sigma_1=iwishrnd(Sigma_hat_1,psi_hat_1);
  thetas_1=randnorm(1,theta_hat_1',chol(Sigma_1./psi_hat_1) )'; 
 
  Sigma_2=iwishrnd(Sigma_hat_2,psi_hat_2);
  thetas_2=randnorm(1,theta_hat_2',chol(Sigma_2./psi_hat_2) )';
  
  while(sum(thetas_1>0)~=s | sum(thetas_2>0)~=s | thetas_1(1) >= thetas_2(1) | thetas_1(2) >= thetas_2(2))
  
  Sigma_1=iwishrnd(Sigma_hat_1,psi_hat_1);
  thetas_1=randnorm(1,theta_hat_1',chol(Sigma_1./psi_hat_1) )'; 
 
  Sigma_2=iwishrnd(Sigma_hat_2,psi_hat_2);
  thetas_2=randnorm(1,theta_hat_2',chol(Sigma_2./psi_hat_2) )';
   end;
  
  % --------------------------------------------------------------------- %
  % impute latent variables                                               %
  % --------------------------------------------------------------------- %
  etas = X*betas_0; % linear predictor in logistic model, inverse cdf methods;
  gs(zs==1) = norminv(unifrnd(normcdf(0,etas(zs==1),sds(zs==1)),ones(sum(zs),1)),etas(zs==1),sds(zs==1)); 
  gs(zs==0) = norminv(unifrnd(zeros(sum(1-zs),1),normcdf(0,etas(zs==0),sds(zs==0))),etas(zs==0),sds(zs==0));
  gs(isinf(gs)==1)=.000001*zs(isinf(gs)==1);  %???
  
  % --------------------------------------------------------------------- %
  % Impute zs
  % -------------------------------------------------------------------- %
  prob  = 1-normcdf(0,etas,sds); % prob of z_i=1
  prop1 = prob.*mvnpdf(Y,thetas_1,Sigma_1);
  prop2 = (1-prob).*mvnpdf(Y,thetas_2,Sigma_2);
  post.prob = prop1./(prop1+prop2);
  
  zs=binornd(1,post.prob);% impute zs from bernoulli
    
  
  % --------------------------------------------------------------------- %
  % update phis and sds                                                   %
  % --------------------------------------------------------------------- %
  phis = gamrnd((nu + 1)/2, 2./(nu + sigma^(-2)*(gs-etas).^2));
  sds = sigma*phis.^(-0.5); 
  %pdi = sds.^(-2);

  if sum(isnan(phis))>0
      break
  end
  
  % --------------------------------------------------------------------- %
  % Update intercept gamma and betas'
  % -------------------------------------------------------------------- %
  SD=repmat(1./sds,[1 p+1]);  
  V_beta=inv((SD.*X)'*(SD.*X)+iLambda); % SD is not iGamma
  E_beta=V_beta*((SD.*X)'*(gs./sds)+iLambda*mus_0);
  V_beta_chol=chol(V_beta);
  betas_0=randnorm(1,E_beta,V_beta_chol); %function from light speed package
  betas=betas_0(2:p+1);
  
  if sum(isnan(betas_0))>0
      break
  end
  
  % --------------------------------------------------------------------- %
  % Update mixing parameters lambda's 
  % -------------------------------------------------------------------- % 
  par1=sqrt(taus./(betas-mus).^2);
  lambdas=1./igrnd(par1,taus,p); % my_fns: igrand by bin
  lambdas(lambdas<10^(-6))=10^(-6);
  lambdas(lambdas>10^(6))=10^(6);  
  iLambda=inv(diag([cv0,lambdas']));  

  
  if sum(isnan(lambdas))>0
      break
  end
  
  % --------------------------------------------------------------------- %
  % Update mus_star and taus_star
  % -------------------------------------------------------------------- %  
  mus_star=zeros(1,p_star);
  %tau2old=taus_star;
  taus_star=ones(1,p_star);
  %muS2=mus_0(2:p+1);
  
  taus_star(1)=gamrnd(ms(1)+a0,1/(sum(lambdas(ks==1)/2)+1/b0));
  
  for j=2:p_star
      taus_star(j)=gamrnd(ms(j)+a1,1/(sum(lambdas(ks==j)/2)+1/b1));
      
      vztemp=1/(sum(1./lambdas(ks==j))+1/d);
      eztemp=vztemp*(sum(betas(ks==j)./lambdas(ks==j))+c/d);
      mus_star(j)=normrnd(eztemp,sqrt(vztemp));
  end
  mus_star(ms==0)=normrnd(c,sqrt(d));
  taus_star(ms==0)=gamrnd(a1,b1);

  if sum(mus_star>100)>0 %???
      break
  end
  
  if sum(isnan(mus_star))>0
      break
  end
  
  if sum(isnan(taus_star))>0
      break
  end
  
  % --------------------------------------------------------------------- %
  % Update V and pi
  % -------------------------------------------------------------------- %
  Vs=zeros(1,p_star);
  pis=zeros(1,p_star);
  Vs(1)=betarnd(ms(1)+1,p-ms(1)+a);
  pis(1)=Vs(1);
  for j=2:p_star
      Vs(j)=betarnd(ms(j)+1,p-sum(ms(1:j))+a);
      pis(j)=(1-sum(pis(1:(j-1))))*Vs(j);
  end
  
  % --------------------------------------------------------------------- %
  % Update configuration ks
  % -------------------------------------------------------------------- %
  Q=zeros(p,p_star);  % proposal density
  for j=1:p_star
%       Q(:,j)=log(pis(j))-1/2*log(s2g(j)')-1/2*(betas_0(2:p+1)-mus_star(j)).^2./s2g(j)';
      Q(:,j)=log(pis(j))-1/2*log(lambdas)-1./(2*lambdas).*(betas-mus_star(j)).^2 + log(taus_star(j)/2)-lambdas*taus_star(j)/2;
  end
  maxQ=max(Q,[],2); %row max
  Q=exp(Q-maxQ(:,ones(1,p_star)));
  M=max((Q./pis(ones(p,1),:)),[],2);
  C=sum(Q,2)+M*(1-sum(pis(1:p_star)));  %normalizing constant
  Qn=Q./C(:,ones(1,p_star));   %normalized prob matrix
  Qcum=zeros(p,p_star+1);  %cumulative prob matrix
  for j=1:p_star
      Qcum(:,j+1)=sum(Qn(:,1:j),2);
  end
  U=unifrnd(0,1,[p, 1]);
  drawnew=sum(U>Qcum(:,p_star+1));  %need to break more sticks?
  maxold = p_star; % same the old max for old configuration ks;
  
  while drawnew>0
      p_star=p_star+1;      
      mus_star(p_star)=normrnd(c,sqrt(d),1);      
      taus_star(p_star)=gamrnd(a1,b1);
      Vs(p_star)=betarnd(1,a);
      pis(p_star)=(1-sum(pis(1:(p_star-1))))*Vs(p_star);
      Q(:,p_star)=M.*pis(p_star).*(U>Qcum(:,p_star)); % (12) in 4c 
      Qn=Q./C(:,ones(1,p_star));
      Qcum(:,p_star+1)=sum(Qn(:,1:p_star),2);
      drawnew=sum(U>Qcum(:,p_star+1));  %check to see if more are still needed
  end
  %determine which bin the U's fall into
  bin=zeros(p,1);
  for j=1:p_star
      bin=bin+(U>Qcum(:,j)&U<Qcum(:,j+1))*j;
  end
  
  %  accept/reject new configuration   
  A=zeros(p,1);  %acceptance probability for metropolis hastings step
  St=(ones(p)-eye(p)).*ks(:,ones(1,p))'+diag(bin); %config if accepted: p_star(i,j) e
  maxnew=max(St,[],2); %for st: each row is new configuration, note that there is prime  
  index=1:p_star;
  Index=index(ones(1,p),:);
  Maxbin=maxnew(:,ones(p_star,1));
  Q2=zeros(p,p_star);
  A1=(bin<=maxold & maxnew==maxold);
  A2=(bin<=maxold & maxnew<maxold);
  A3=bin>maxold;
  Prind=Index<=Maxbin;
  Prsum=Prind*pis';

  if sum(A1)~=p
  for j=1:p_star
%       Q2(:,j)=(log(pis(j))-1/2*log(s2g(j)')-1/2*(betas_0(2:p+1)-mus_star(j)).^2./s2g(j)').*(j<=maxnew);     
      Q2(:,j)=(log(pis(j))-1/2*log(lambdas)-1./(2*lambdas).*(betas-mus_star(j)).^2+log(taus_star(j)/2)-lambdas*taus_star(j)/2).*(j<=maxnew);
  end      
  Q2(Q2==0)=-Inf;
  maxQ2=max(Q2,[],2);
  Q2=exp(Q2-maxQ2(:,ones(1,p_star)));
  M2=max((Q2./pis(ones(p,1),:)),[],2);
  C2=sum(Q2,2)+M2.*(1-Prsum);  
  Q2n=Q2./C2(:,ones(1,p_star));
  Bin=bin(:,ones(p_star,1));
  Zind=(Index==Bin);
  Zind2=(Index==ks(:,ones(p_star,1)));
  tmp=(log(Qn)-log(pis(ones(p,1),:)))';
  A=(exp(-(log(C2))+log(M2)-tmp(Zind2'))).*A2+A;
  tmp=(log(Q2n)-log(pis(ones(p,1),:)))';
  A=exp((log(C)+tmp(Zind')-log(M))).*A3+A;
  end
  A=A+A1;
  A(A>1)=1;
  Sw=binornd(1,A,[p 1]);
  ks=bin.*Sw+ks.*(1-Sw);
  p_star=max(ks);
  
  mus=mus_star(ks')';
  mus_0=[c0,mus']';
  taus=taus_star(ks')';
  
  Ssum=(repmat(ks(:,1),[1 p_star])-repmat(1:p_star,[p 1])==0); 
  ms = (ones(1,p)*Ssum)';              %no in each grp.
  
  
  % --------------------------------------------------------------------- %
  % Save outputs
  % -------------------------------------------------------------------- %
  if (g>burnin & rem(g/thin,1)==0)
    %disp([g, p_star,sum(zs)])
    
    betas_0_out((g-burnin)/thin,:)=betas_0';
    %mus_out((g-burnin)/thin,:)=mus';
    %taus_out((g-burnin)/thin,:)=taus';
    
    thetas_out((g-burnin)/thin,:)=[thetas_1,thetas_2];
    
    tmp1=reshape(triu(Sigma_1)',1,s*s);
    tmp2=reshape(triu(Sigma_2)',1,s*s);
    Sigmas_out((g-burnin)/thin,:)=[tmp1(find(tmp1~=0)),tmp2(find(tmp2~=0))];
    
    zs_out((g-burnin)/thin,:)=zs';
  % phis_out((g-burnin)/thin,:)=phis';
   %ks_out((g-burnin)/thin,:)=ks';
   %taus_star_out((g-burnin)/thin,:)=taus_star';
   %lambdas_out((g-burnin)/thin,:)=lambdas';
  end
  
  
  %
  
end
tout=[betas_0_out,thetas_out,Sigmas_out,zs_out];
  %       p+1         2s        s*(s+1)   n
 
%save MCMC output;
%csvwrite('Results\msp_raceSGA_5.csv',tout);
%csvwrite('Results\msp_raceSGA_10.csv',tout);
%csvwrite('\\chgserve5\home\bz27\profile\desktop\tmp\msp_bivariate_test.csv',tout);
%csvwrite('\\chgserve5\home\bz27\profile\desktop\tmp\msp_bivariate_simu2.csv',tout);
%figure(1); clf
%mcmcplot(thetas_out(1:1000,:),[1:4],[],'chainpanel')


