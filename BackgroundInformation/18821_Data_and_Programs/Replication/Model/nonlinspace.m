function x = nonlinspace(lo,hi,n,phi)
    
    % recursively constructs an unequally spaced grid.
    % phi > 1 -> more mass at the lower end of the grid.
    % lo can be a vector (x then becomes a matrix).

    x      = NaN(n,length(lo));
    x(1,:) = lo;
    for i = 2:n
        x(i,:) = x(i-1,:) + (hi-x(i-1,:))./((n-i+1)^phi);
    end

end