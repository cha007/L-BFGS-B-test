function [ alphas, betas, thetas, final_likelihood] = CCNF_training_bfgs(thresholdX, thresholdFun, x, y, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, similarityFNs, sparsityFNs, varargin)
%CCNF_training_bfgs Performs CCNF training using BFGS (or LBFGS)
    %save matfile_x x;
    %save matfile_y y;
    if(sum(strcmp(varargin,'const')))
        ind = find(strcmp(varargin,'const')) + 1;
        const = varargin{ind};        
    else        
        const = false;
    end
    
    if(iscell(x))        
        num_seqs = numel(x);
                
        x = cell2mat(x)';
        % add a bias term
        x =  cat(1, ones(1,size(x,2)), x);        
        
        % If all of the sequences are of the same length can flatten them
        % to the same matrix
        if(const)
            
            y = cell2mat(y);
            y = reshape(y, numel(y)/num_seqs, num_seqs);
        end
        
    else
        % if not a cell it has already been flattened, and is constant
        % (most likely)
        num_seqs = varargin{find(strcmp(varargin, 'num_seqs'))+1};
    end
    
    % Should try a bunch of seed for initialising theta?
    if(sum(strcmp(varargin,'reinit')))
        ind = find(strcmp(varargin,'reinit')) + 1;
        reinit = varargin{ind};
    else
        reinit = false; 
    end
        
    % It is possible to predefine the components B^(k) and C^(k) required 
    % to compute B and and C terms and partial derivative (from equations 
    % 30 and 31 in Appendix B), also can predefine yB^(k)y and yC^(k)y,
    % as they also do not change through the iterations
    % In constant case Precalc_Bs are same across the sequences, same for 
    % PrecalcBsFlat, however yB^(k)y is defined per sequence
    if(sum(strcmp(varargin,'PrecalcBs')) && sum(strcmp(varargin,'PrecalcBsFlat'))...
             && sum(strcmp(varargin,'Precalc_yBy')))
        ind = find(strcmp(varargin,'PrecalcBs')) + 1;
        Precalc_Bs = varargin{ind};

        ind = find(strcmp(varargin,'PrecalcBsFlat')) + 1;
        Precalc_Bs_flat = varargin{ind};

        ind = find(strcmp(varargin,'Precalc_yBys')) + 1;
        Precalc_yBys = varargin{ind};
    else
        % if these are not provided calculate them        
        [ ~, Precalc_Bs, Precalc_Bs_flat, Precalc_yBys ] = CalculateSimilarities( num_seqs, x, similarityFNs, sparsityFNs, y, const);
      
        Precalc_Bs_1 = Precalc_Bs{1}{1}; %save matfile_Precalc_Bs_0        Precalc_Bs_1;
        Precalc_Bs_2 = Precalc_Bs{1}{2}; %save matfile_Precalc_Bs_1        Precalc_Bs_2;
        Precalc_Bs_3 = Precalc_Bs{1}{3}; %save matfile_Precalc_Bs_2        Precalc_Bs_3;
        Precalc_Bs_flat_1 = Precalc_Bs_flat{1};  %save matfile_Precalc_Bs_flat    Precalc_Bs_flat_1;
        %save matfile_Precalc_yBys       Precalc_yBys;
    end
                    
    % Reinitialisation attempts to find a better starting point for the
    % model training (sometimes helps sometimes doesn't)
    if(reinit)
        
        rng(0);
        
        % By default try 200 times, but can override
        num_reinit = 200;
        
        if(sum(strcmp(varargin,'num_reinit')))
            num_reinit = varargin{find(strcmp(varargin,'num_reinit')) + 1};
        end
        
        thetas_good = cell(num_reinit, 1);
        lhoods = zeros(num_reinit, 1);
        for i=1:num_reinit
            initial_Theta = randInitializeWeights(size(thetas,2)-1, numel(alphas));            
            lhoods(i) = LogLikelihoodCCNF(y, x, alphas, betas, initial_Theta, lambda_a, lambda_b, lambda_th, Precalc_Bs_flat, [], [], [], [], const, num_seqs);
            thetas_good{i} = initial_Theta;
        end
        [~,ind_max] = max(lhoods);
        thetas = thetas_good{ind_max};
    end

    params = [alphas; betas; thetas(:)];
    %save matfile_params params;
    
    if(any(strcmp(varargin,'lbfgs')))
        options = optimset('Algorithm','interior-point','GradObj','on', 'Hessian', 'lbfgs', 'TolX', thresholdX, 'TolFun', thresholdFun, 'display', 'on');
    else
        options = optimset('Algorithm','interior-point','GradObj','on', 'Hessian', 'bfgs', 'TolX', thresholdX, 'TolFun', thresholdFun, 'display', 'off');
    end
    
    if(any(strcmp(varargin,'max_iter')))
        options.MaxIter = varargin{find(strcmp(varargin,'max_iter')) + 1};
    end
    
    objectiveFun = @(params)objectiveFunction(params, numel(alphas), numel(betas), size(thetas), lambda_a, lambda_b, lambda_th, Precalc_Bs, x, y, Precalc_yBys, Precalc_Bs_flat, const);
    lowerBound = [zeros(numel(alphas)+numel(betas),1); -Inf(numel(thetas),1)];
    upperBound = Inf(numel(params),1);
    
 %   tic;
 %   Opt = opti('fun',objectiveFun,'x0',params,'bounds',lowerBound,upperBound);
  %  [x,fval] = solve(Opt);
 %   toc;
   
    tic;
    %options.UseParallel = true;
    %problem = createOptimProblem('fmincon','objective',objectiveFun,'x0',params,'lb',lowerBound,'ub',upperBound,'options',options); %x is the variable I wish to min/max
    %ms = MultiStart;
    %params = run(ms,problem,1);
    
    %options.UseParallel = true;
    %params = fmincon(objectiveFun, params, [], [],[],[], lowerBound, upperBound, [], options);
    %aa = params';
   %save 'test.txt' aa -ascii;
   
   params2 = params;
   [params2 loss] = ccnf_test(params2,Precalc_Bs_1,Precalc_Bs_2,Precalc_Bs_3,x,y,Precalc_yBys,Precalc_Bs_flat{1});
   %[params, fval,exitflag,output] = fmincon(objectiveFun, params, [], [],[],[], lowerBound, upperBound, [], options);
    alphas = params2(1:numel(alphas));
    betas = params2(numel(alphas)+1:numel(alphas)+numel(betas));
    thetas = reshape(params2(numel(alphas) + numel(betas) + 1:end), size(thetas));
    
    final_likelihood = LogLikelihoodCCNF(y, x, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, Precalc_Bs_flat, [], [], [], [], const, num_seqs);
    %sprintf('cuda test final_likelihood %f\n', final_likelihood)

   %tic;
   %[params, fval,exitflag,output] = fmincon(objectiveFun, params, [], [],[],[], lowerBound, upperBound, [], options);
   %toc;
   
    %alphas = params(1:numel(alphas));
    %betas = params(numel(alphas)+1:numel(alphas)+numel(betas));
    %thetas = reshape(params(numel(alphas) + numel(betas) + 1:end), size(thetas));
    
    %final_likelihood = LogLikelihoodCCNF(y, x, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, Precalc_Bs_flat, [], [], [], [], const, num_seqs);
    %sprintf('final_likelihood %f\n', final_likelihood)
    %sprintf('final\n')
end

function [loss, gradient] = objectiveFunction(params, numAlpha, numBeta, sizeTheta, lambda_a, lambda_b, lambda_th, PrecalcQ2s, x, y, PrecalcYqDs, PrecalcQ2sFlat, const)
    alphas = params(1:numAlpha);
    betas = params(numAlpha+1:numAlpha+numBeta);
    thetas = reshape(params(numAlpha + numBeta + 1:end), sizeTheta);
    
    num_seqs = size(PrecalcYqDs,1);
    aa = params';   %save 'test.txt' aa -ascii -append;
    [gradient, SigmaInvs, CholDecomps, Sigmas, bs, all_x_resp] = gradientCCNF(params, numAlpha, numBeta, sizeTheta, lambda_a, lambda_b, lambda_th, PrecalcQ2s, x, y, PrecalcYqDs, PrecalcQ2sFlat, const, num_seqs);
    aa = gradient';   %save 'test.txt' aa -ascii -append;
    % as bfgs does gradient descent rather than ascent, negate the results
    gradient = -gradient;
    loss = -LogLikelihoodCCNF(y, x, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, PrecalcQ2sFlat, SigmaInvs, CholDecomps, Sigmas, bs, const, num_seqs, all_x_resp);
    %sprintf('test %f\n', loss)
end
