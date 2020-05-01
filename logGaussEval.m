function p = logGaussEval(x, mu, P) 
%gaussEval evaulates a gaussian distribution with mean mu, covariance P,
%at x

p = log(1/sqrt((2*pi)^length(x) * det(P))) + (-.5*(x - mu)'/P*(x - mu));
% p = (-.5*(x - mu)'/P*(x - mu));

if(isnan(p))
    p = 0;
end

end