function [out1, out2, out3]=goldensectionsearchvar(funfcn, a, b, m, maxiter,ftol,varargin)


%This function computes minima with a simultaneous line search (golden section) for a vectorized,
%parametrized function |R_1xm into |R, where m is a finite number of
%parameters. It is a variant of goldensectionsearch.m, that differs from
%the former by specifying the first midpoint m, instead of setting m equal
%to the golden section between a and b. All that is required of m now is
%that it has a smaller value than at least one of the brackets, not both!

%Inputs:              funfcn:   The function name to be evaluated
%                     a:        One side of the bracketing initial conditions
%                     b:        Second side of the bracketing initial conditions
%                     m:        The midpoint.    
%                     maxiter:  The maximum number of iterations allowed.
%                     ftol:     A stopping criterion, approx. 0.
%                     varargin: The parameters for the function to be
%                               evaluated.

%Outputs:             out1:     The argminimum of the function in the interval
%                               [x1, x2].
%                     out2:     The minimum of the function in the interval
%                               [x1, x2].
%                     out3:     The number of iterations used.

%Check Specifications


if isequal(size(a),size(b))==0 | isequal(size(a),size(m))==0
    error('Bracketing vectors and midpoint must be of the same length.')
end




checkbracketing=((b-a)>0);
if any(checkbracketing==0)
    error('Lower and upper interval points are not in the correct sequence.')
end

checkbracketing=((b-m)>0);
if any(checkbracketing==0)
    error('Upper interval points and the midpoints are not in the correct sequence.')
end

checkbracketing=((m-a)>0);
if any(checkbracketing==0)
    error('Lower interval points and the midpoints are not in the correct sequence.')
end

%Initial Set-Up
lambda=(3-sqrt(5))*inv(2);       %the golden section constant

index=((b-m)>(m-a));
zgs=(lambda*m+(1-lambda)*b).*index+(1-index).*(lambda*a+(1-lambda)*m); %new trial point

%Values at all four points
fa=feval(funfcn,a,varargin{:});
fb=feval(funfcn,b,varargin{:});
fm=feval(funfcn,m,varargin{:});

checkgoldensection=(fm<max(fa,fb));
if any(checkgoldensection==0)
    error('Golden Section Search started with bad values: the golden section has a larger value than both endpoints.')
end

fzgs=feval(funfcn,zgs,varargin{:});

%Start Loop for the Golden Section Search
metric=10^3;
iter=1;    
while metric>ftol & iter<=maxiter
    a=a.*index.*(fzgs>fm)+a.*(1-index).*(fm>fzgs)+m.*index.*(fzgs<=fm)+zgs.*(1-index).*(fm<=fzgs);
    b=zgs.*index.*(fzgs>fm)+m.*(1-index).*(fm>fzgs)+b.*index.*(fzgs<=fm)+b.*(1-index).*(fm<=fzgs);
    m=m.*index.*(fzgs>fm)+zgs.*(1-index).*(fm>fzgs)+zgs.*index.*(fzgs<=fm)+m.*(1-index).*(fm<=fzgs);
    index=((b-m)>(m-a));
    zgs=(lambda*m+(1-lambda)*b).*index+(1-index).*(lambda*a+(1-lambda)*m);
    fm=feval(funfcn,m,varargin{:});
    fzgs=feval(funfcn,zgs,varargin{:});
    metric=max(abs(fm-fzgs));
    iter=iter+1;
end

%Results
out1=zgs;
out2=fzgs;
out3=iter-1;

