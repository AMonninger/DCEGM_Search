function f = trapint(x,dim)
% trapezoidal integration

f = x;

for kk=1:length(dim)
    Ndim = length(dim{kk});
    if Ndim>1 
        f = trapz(dim{kk},reshape(f,[Ndim,numel(f)/Ndim]),1);
    end
end

end