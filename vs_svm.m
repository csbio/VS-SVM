function vs_svm()

% (ligand-based) Virtual Screening using SVM

% License: 
% This machine learning model script is free for academic use; 
% Contact hamid@umn.edu or chadm@umn.edu for any commercial use.

% Citation:
% Hamid Safizadeh, et al. Improving Measures of Chemical Structural Similarity
% Using Machine Learning on Chemical-Genetic Interactions. Journal of Chemical
% Information and Modeling, 2021.

%% Initialization

path = genpath('LibSVM');
addpath(path);

path = genpath('PCA_Supervised');
addpath(path);

fp_depth = 'ASP-8';

% Note: A slash is required at the beginning only.
fld_model = '/Learning_Model';

% Note: Slashes are required at the beginning and the end.
fld_gold = '/CG_Gold/';

LOG = fopen([fld_model '/log.txt'],'w');

%% SVM Regression

start = 1; % starting bootstrap number
N = 200; % ending bootstrap number (number of bootstraps)
thr = 95; % percentage of the explained variance in S-PCA
frg_thr = zeros(N,1); % number of the required S-PCs
prd_inv = -10; % Invalid initial "prd"

% Training and test compound indices
cpds_trn_ord = [];
cpds_tst_ord = [];

% Training and test predictions
prd_trn_bts = [];
prd_tst_bts = [];

% Each experiment corresponds to one round of bootstrapping.
for expr = start : N 
    fprintf('Experiment %d\n\n',expr);
    fprintf(LOG,'Experiment %d\n\n',expr);
    
    load([fld_model '/Cpds_Ref_Bts.mat'])
    load([fld_gold fp_depth '/Frags_Ord_Spr.mat'])
    load([fld_gold fp_depth '/Profs_New.mat'])
    load([fld_gold fp_depth '/Cpds_Prf_New.mat'])
    
    % Removing Artemisinins (a cluster of 20 compounds)
    load([fld_gold fp_depth '/Artemisinins.mat'])
    frags_ord(idx_cpds_artm,:) = [];
    profs(idx_cpds_artm,:) = [];
    cpds_prf(idx_cpds_artm,:) = [];
    
    if expr==1
        save([fld_model '/Prd_Bts.mat'],'prd_trn_bts','prd_tst_bts')
    end
    
    if expr>1
        load([fld_model '/Cpds_Trn_Tst.mat'])
    end
    
    cpds_trn_bin = zeros(size(cpds_prf,1),1); % final binary indices
    cpds_tst_bin = zeros(size(cpds_prf,1),1); % final binary indices

    idx_trn_bin = cpds_trn_bts(:,expr); % primary binary indices
    idx_tst_bin = cpds_tst_bts(:,expr); % primary binary indices
    
    % Synchronization of indices of available compounds
    [~,idx_sync] = ismember(cpds_ref,cpds_prf,'rows');
    idx_trn = idx_sync(find(idx_trn_bin));
    idx_tst = idx_sync(find(idx_tst_bin));
    idx_trn(find(idx_trn==0)) = [];
    idx_tst(find(idx_tst==0)) = [];
    
    clearvars('cpds_trn_bts','cpds_tst_bts','cpds_ref')
    
    % Sorting training and test indices
    idx_trn = sort(idx_trn);
    idx_tst = sort(idx_tst);
    
    % Creating training and test data
    profs_trn = profs(idx_trn,:);
    profs_tst = profs(idx_tst,:);
    frags_trn = frags_ord(idx_trn,:);
    frags_tst = frags_ord(idx_tst,:);
    
    % Finding all-zero training compounds in the original dataset
    idx_zero_trn = find(all(frags_trn==0,2));
    profs_trn(idx_zero_trn,:) = [];
    frags_trn(idx_zero_trn,:) = [];
    idx_trn(idx_zero_trn) = [];
        
    % Note: The original data refers to the data before applying S-PCA.
    fprintf('Number of all-zero compounds (org, trn): %d\n',length(idx_zero_trn));
    fprintf('Total number of compounds (org, trn): %d\n',size(frags_trn,1));
    fprintf(LOG,'Number of all-zero compounds (org, trn): %d\n',length(idx_zero_trn));
    fprintf(LOG,'Total number of compounds (org, trn): %d\n',size(frags_trn,1));
    
    % Finding all-zero test compounds in the original dataset 
    idx_zero_tst = find(all(frags_tst==0,2));
    profs_tst(idx_zero_tst,:) = [];
    frags_tst(idx_zero_tst,:) = [];
    idx_tst(idx_zero_tst) = [];
    
    fprintf('Number of all-zero compounds (org, tst): %d\n',length(idx_zero_tst));
    fprintf('Total number of compounds (org, tst): %d\n',size(frags_tst,1));
    fprintf(LOG,'Number of all-zero compounds (org, tst): %d\n',length(idx_zero_tst));
    fprintf(LOG,'Total number of compounds (org, tst): %d\n',size(frags_tst,1));

    fprintf('Supervised PCA...');
    fprintf(LOG,'Supervised PCA...');
    
    param.ktype_y = 'linear';
    param.kparam_y = 1;
    [z_trn,U,D] = SPCA(transpose(frags_trn),transpose(profs_trn),length(idx_trn),param);
    
    explained_cum = cumsum(D)/sum(D)*100;
    frg_thr_expr = find(explained_cum>=thr,1);
        
    frags_trn = transpose(z_trn(1:frg_thr_expr,:));
    frags_tst = frags_tst*U(:,1:frg_thr_expr);
    
    clearvars('frags_ord','profs','cpds_prf')
    
    fprintf(' Features reduced to %d S-PCs.\n',frg_thr_expr);
    fprintf(LOG,' Features reduced to %d S-PCs.\n',frg_thr_expr);
        
    % Finding all-zero training compounds after S-PCA
    idx_zero_trn = find(all(frags_trn==0,2));
    profs_trn(idx_zero_trn,:) = [];
    frags_trn(idx_zero_trn,:) = [];
    idx_trn(idx_zero_trn) = [];
        
    % Note: The reduced data refers to the data after applying S-PCA.
    fprintf('Number of all-zero compounds (rdc, trn): %d\n',length(idx_zero_trn));
    fprintf('Total number of compounds (rdc, trn): %d\n',size(frags_trn,1));
    fprintf(LOG,'Number of all-zero compounds (rdc, trn): %d\n',length(idx_zero_trn));
    fprintf(LOG,'Total number of compounds (rdc, trn): %d\n',size(frags_trn,1));
    
    % Finding all-zero test compounds after S-PCA 
    idx_zero_tst = find(all(frags_tst==0,2));
    profs_tst(idx_zero_tst,:) = [];
    frags_tst(idx_zero_tst,:) = [];
    idx_tst(idx_zero_tst) = [];
    
    fprintf('Number of all-zero compounds (rdc, tst): %d\n',length(idx_zero_tst));
    fprintf('Total number of compounds (rdc, tst): %d\n',size(frags_tst,1));
    fprintf(LOG,'Number of all-zero compounds (rdc, tst): %d\n',length(idx_zero_tst));
    fprintf(LOG,'Total number of compounds (rdc, tst): %d\n',size(frags_tst,1));
    
    cpds_trn_bin(idx_trn) = 1;
    cpds_trn_ord = [cpds_trn_ord cpds_trn_bin];
    
    cpds_tst_bin(idx_tst) = 1;
    cpds_tst_ord = [cpds_tst_ord cpds_tst_bin];
    
    frg_thr(expr) = frg_thr_expr;
        
    % Indices of training and test pairwise combinations

    fprintf('Computing indices of pairwise combinations...\n');
    fprintf(LOG,'Computing indices of pairwise combinations...\n');

    comb_trn = nchoosek(1:length(idx_trn),2);
    comb_tst = nchoosek(1:length(idx_tst),2);
    
    fprintf('Computing pairwise combinations for training...\n');
    fprintf(LOG,'Computing pairwise combinations for training...\n');

    % All possible pairwise combinations (training)
    frags1 = sparse(frags_trn(comb_trn(:,1),:));
    frags2 = sparse(frags_trn(comb_trn(:,2),:));

    frags1 = bsxfun(@rdivide,frags1,sqrt(sum(abs(frags1).^2,2)));
    frags2 = bsxfun(@rdivide,frags2,sqrt(sum(abs(frags2).^2,2)));
    frg_trn = bsxfun(@times,frags1,frags2);
    clearvars('frags_trn','frags1','frags2')

    fprintf('Min/Max training frags: %f and %f\n',...
        full(min(min(frg_trn))),full(max(max(frg_trn))));
    fprintf(LOG,'Min/Max training frags: %f and %f\n',...
        full(min(min(frg_trn))),full(max(max(frg_trn))));

    fprintf('Computing pairwise cosine similarities for training...\n');
    fprintf(LOG,'Computing pairwise cosine similarities for training...\n');

    % Pairwise cosine similarities (training)
    sim_profs = sim_profs_cos(profs_trn);
    sim_diag_zero = sim_profs-diag(diag(sim_profs));
    sim_trn = squareform(sim_diag_zero);
    clearvars('profs_trn','sim_profs','sim_diag_zero')
    
    fprintf('Computing pairwise combinations for test...\n');
    fprintf(LOG,'Computing pairwise combinations for test...\n');

    % All possible pairwise combinations (test)
    frags1 = sparse(frags_tst(comb_tst(:,1),:));
    frags2 = sparse(frags_tst(comb_tst(:,2),:));

    frags1 = bsxfun(@rdivide,frags1,sqrt(sum(abs(frags1).^2,2)));
    frags2 = bsxfun(@rdivide,frags2,sqrt(sum(abs(frags2).^2,2)));
    frg_tst = bsxfun(@times,frags1,frags2);
    clearvars('frags_tst','frags1','frags2')

    fprintf('Min/Max test frags: %f and %f\n',...
        full(min(min(frg_tst))),full(max(max(frg_tst))));
    fprintf(LOG,'Min/Max test frags: %f and %f\n',...
        full(min(min(frg_tst))),full(max(max(frg_tst))));
    
    fprintf('Computing pairwise cosine similarities for test...\n');
    fprintf(LOG,'Computing pairwise cosine similarities for test...\n');

    % Pairwise cosine similarities (test)
    sim_profs = sim_profs_cos(profs_tst);
    sim_diag_zero = sim_profs-diag(diag(sim_profs)); 
    sim_tst = squareform(sim_diag_zero);
    clearvars('profs_tst','sim_profs','sim_diag_zero')
    
    % Creating a learning model based on the whole training set may not be feasible!
    % Random selection of the training points (# original training points > 14M)
    
    rnd_trn = 1.0; % a percentage for random selection (x100)
    num_trn = length(sim_trn); % # original training points
    sub_trn = sort(randperm(num_trn,ceil(num_trn*rnd_trn)));

    frg_trn = frg_trn(sub_trn,:);
    sim_trn = sim_trn(sub_trn);

    fprintf('Reduced number of compound pairs in the training set: %g\n',length(sub_trn));
    fprintf(LOG,'Reduced number of compound pairs in the training set: %g\n',length(sub_trn));
    
    % Evaluating the learning model based on the whole test set may not be feasible!
    % Random selection of the test points (# original test points > 14M)
    
    rnd_tst = 1.0; % a percentage for random selection (x100)
    num_tst = length(sim_tst); % # original training points
    sub_tst = sort(randperm(num_tst,ceil(num_tst*rnd_tst)));

    frg_tst = frg_tst(sub_tst,:);
    sim_tst = sim_tst(sub_tst);

    fprintf('Reduced number of compound pairs in the test set: %g\n',length(sub_tst));
    fprintf(LOG,'Reduced number of compound pairs in the test set: %g\n',length(sub_tst));
    
    fprintf('Saving general matrices...\n');
    fprintf(LOG,'Saving general matrices...\n');
    
    % Separate fld_model for each experiment
    fld_expr = [fld_model '/Btsp' num2str(expr)];
    mkdir(fld_expr);
    
    suffix = ['Btsp' num2str(expr)];
    
    % Saving general results
    save([fld_model '/Cpds_Trn_Tst.mat'],'cpds_trn_ord','cpds_tst_ord','frg_thr')
    save([fld_expr '/Frags_Pair_Trn_' suffix '.mat'],'frg_trn','sim_trn','-v7.3')
    save([fld_expr '/Frags_Pair_Tst_' suffix '.mat'],'frg_tst','sim_tst','-v7.3')
    save([fld_expr '/Frags_Sub_Trn_' suffix '.mat'],'sub_trn')
    save([fld_expr '/Frags_Sub_Tst_' suffix '.mat'],'sub_tst')
    
    fprintf('SVM regression...\n');
    fprintf(LOG,'SVM regression...\n');

    sim_trn = transpose(sim_trn);
    % sim_tst = transpose(sim_tst);

    % epsilon-SVR
    param.s = 3;
    % RBF kernel
    param.t = 2;
    % cost parameter
    param.cset = 0.5;
    % degree parameter
    param.dset = 2; % It does not matter for the RBF kernel.
    % gamma parameter
    param.gset = 300; 
    % epsilon parameter
    param.eset = 0.3;
    % nfold cross-validation
    param.nfold = 1;
    % shrinking parameter
    param.h = 0;
    % cache-size parameter (default: 100 MB)
    param.m = 15000;

    cor_prd_rdc_tst_all = zeros(length(param.gset),length(param.cset));
    cor_bb_org_tst_all = zeros(length(param.gset),length(param.cset));
    cor_cos_rdc_tst_all = zeros(length(param.gset),length(param.cset));

    for e = 1 : length(param.eset)
        for d = 1 : length(param.dset) % length(param.dset) has to be 1.
            for c = 1 : length(param.cset)            
                for g = 1 : length(param.gset)
                    if length(param.dset)>1
                        fprintf('\nError: Polynomial degree>1!\n');
                    end

                    param.c = param.cset(c);
                    param.d = param.dset(d);
                    param.e = param.eset(e);
                    param.g = param.gset(g);

                    fprintf('\nCurrent evaluation: c = %g, g = %g, e = %g\n',param.c,param.g,param.e);
                    fprintf(LOG,'\nCurrent evaluation: c = %g, g = %g, e = %g\n',param.c,param.g,param.e);

                    param.libsvm = ['-s ',num2str(param.s),' -t ',num2str(param.t),' -c ',num2str(param.c),...
                        ' -d ',num2str(param.d), ' -g ',num2str(param.g),' -p ',num2str(param.e),...
                        ' -h ',num2str(param.h),' -m ',num2str(param.m)];
                    
                    clearvars('frg_tst','sim_tst')

                    model = svmtrain(sim_trn,frg_trn,param.libsvm);

                    nu = model.totalSV/size(frg_trn,1);
                    fprintf('#SVs: %d, #Samples: %d, SVs%%: %7.5f\n',model.totalSV,size(frg_trn,1),nu*100); 
                    fprintf(LOG,'#SVs: %d, #Samples: %d, SVs%%: %7.5f\n',model.totalSV,size(frg_trn,1),nu*100);
                    
                    % prd_trn = svmpredict(sim_trn,frg_trn,model);
                    
                    % Not needed in the current experiment anymore.
                    clearvars('frg_trn')
                    
                    load([fld_expr '/Frags_Pair_Tst_' suffix '.mat'])
                    sim_tst = transpose(sim_tst);
                    
                    prd_tst = svmpredict(sim_tst,frg_tst,model);                       

                    % fprintf('MSE (training): %10.8f, MSE (test): %10.8f\n',mean((prd_trn-sim_trn).^2),mean((prd_tst-sim_tst).^2));
                    % fprintf(LOG,'MSE (training): %10.8f, MSE (test): %10.8f\n',mean((prd_trn-sim_trn).^2),mean((prd_tst-sim_tst).^2));

                    fprintf('MSE (test): %10.8f\n',mean((prd_tst-sim_tst).^2));
                    fprintf(LOG,'MSE (test): %10.8f\n',mean((prd_tst-sim_tst).^2));
                    
                    % This will be used for "correlation of cosine test similarities (reduced)".
                    sim_cos_tst = sum(frg_tst,2);
                    
                    % Not needed in the current experiment anymore.
                    clearvars('frg_tst')

                    % Expansion of bootstrapping matrices
                    load([fld_gold fp_depth '/Frags_Ord_Spr.mat'])
                    load([fld_model '/Prd_Bts.mat'])
                    frags_ord(idx_cpds_artm,:) = []; % Removing Artemisinins
                    
                    comb_frg_ord = nchoosek(1:size(frags_ord,1),2);
                    % comb_trn_bts = nchoosek(idx_trn,2);
                    comb_tst_bts = nchoosek(idx_tst,2);
                    
                    % [~,idx_comb_trn] = ismember(comb_trn_bts,comb_frg_ord,'rows');
                    % prd_trn_expr = ones(size(comb_frg_ord,1),1)*prd_inv;
                    % prd_trn_expr(idx_comb_trn(sub_trn)) = prd_trn;
                    % prd_trn_bts = [prd_trn_bts prd_trn_expr];
                    
                    [~,idx_comb_tst] = ismember(comb_tst_bts,comb_frg_ord,'rows');
                    prd_tst_expr = ones(size(comb_frg_ord,1),1)*prd_inv;
                    prd_tst_expr(idx_comb_tst(sub_tst)) = prd_tst;
                    prd_tst_bts = [prd_tst_bts prd_tst_expr];
                    
                    clearvars('comb_trn_bts','prd_trn_expr')
                    clearvars('comb_frg_ord','comb_tst_bts','prd_tst_expr')
                    
                    % Correlation of predicted test similarities (reduced)
                    [cor,p_val] = corrcoef(full(prd_tst),full(sim_tst));
                    cor_prd_rdc_tst = cor(2,1);
                    p_val_prd_rdc_tst = p_val(2,1);
                    cor_prd_rdc_tst_all(g,c) = cor(2,1);

                    fprintf('Cor/P-val (test, prd, rdc): %6.4f and %g\n',cor_prd_rdc_tst,p_val_prd_rdc_tst);
                    fprintf(LOG,'Cor/P-val (test, prd, rdc): %6.4f and %g\n',cor_prd_rdc_tst,p_val_prd_rdc_tst);

                    % Correlation of braun-blanquet test similarities (original)
                    frags_tst = frags_ord(idx_tst,:); % Original space

                    % Pairwise braun-blanquet test similarities (original)
                    sim_frags = sim_frags_bb(frags_tst);
                    sim_diag_zero = sim_frags-diag(diag(sim_frags)); 
                    sim_bb_tst = squareform(sim_diag_zero);

                    [cor,p_val] = corrcoef(full(sim_bb_tst),full(sim_tst));
                    cor_bb_org_tst = cor(2,1);
                    p_val_bb_org_tst = p_val(2,1);
                    cor_bb_org_tst_all(g,c) = cor(2,1);

                    fprintf('Cor/P-val (test, bb, org):  %6.4f and %g\n',cor_bb_org_tst,p_val_bb_org_tst);
                    fprintf(LOG,'Cor/P-val (test, bb, org):  %6.4f and %g\n',cor_bb_org_tst,p_val_bb_org_tst);

                    % Correlation of cosine test similarities (reduced)
                    [cor,p_val] = corrcoef(full(sim_cos_tst),full(sim_tst));
                    cor_cos_rdc_tst = cor(2,1);
                    p_val_cos_rdc_tst = p_val(2,1);
                    cor_cos_rdc_tst_all(g,c) = cor(2,1);

                    fprintf('Cor/P-val (test, cos, rdc): %6.4f and %g\n',cor_cos_rdc_tst,p_val_cos_rdc_tst);
                    fprintf(LOG,'Cor/P-val (test, cos, rdc): %6.4f and %g\n',cor_cos_rdc_tst,p_val_cos_rdc_tst);
                    
                    clearvars('frags_ord','frags_tst')
                    
                    save([fld_expr '/Model_SVM_' suffix '_c' num2str(c) '_g' num2str(g) '_e' num2str(e) '.mat'],...
                        'model','param','nu','prd_tst') % ,'prd_trn')
                    save([fld_expr '/Model_Cor_' suffix '_c' num2str(c) '_g' num2str(g) '_e' num2str(e) '.mat'],...
                        'cor_prd_rdc_tst','p_val_prd_rdc_tst','cor_bb_org_tst','p_val_bb_org_tst','cor_cos_rdc_tst','p_val_cos_rdc_tst')
                    save([fld_expr '/Model_Cor_' suffix '_all.mat'],...
                        'cor_prd_rdc_tst_all','cor_bb_org_tst_all','cor_cos_rdc_tst_all')
                end
            end
        end
    end

%     % Average of the available training bootstraps
%     msk_prd_trn = prd_trn_bts~=prd_inv;
%     cnt_prd_trn = sum(msk_prd_trn,2);
%     cnt_prd_trn(find(cnt_prd_trn==0)) = 1;
%     val_prd_trn = prd_trn_bts.*msk_prd_trn;
%     prd_trn = sum(val_prd_trn,2)./cnt_prd_trn;
    
    % Average of the available testing bootstraps
    msk_prd_tst = prd_tst_bts~=prd_inv;
    cnt_prd_tst = sum(msk_prd_tst,2);
    cnt_prd_tst(find(cnt_prd_tst==0)) = 1;
    val_prd_tst = prd_tst_bts.*msk_prd_tst;
    prd_tst = sum(val_prd_tst,2)./cnt_prd_tst;
    
    save([fld_model '/Prd_Bts.mat'],'prd_tst_bts','-v7.3') % ,'prd_trn_bts')
    save([fld_model '/Prd_Avg.mat'],'prd_tst','prd_inv') % ,'prd_trn')
    
    % clearvars('prd_trn_bts','prd_trn')
    clearvars('prd_tst_bts','prd_tst')
   
    fprintf('\n\n');
    fprintf(LOG,'\n\n');
end

end