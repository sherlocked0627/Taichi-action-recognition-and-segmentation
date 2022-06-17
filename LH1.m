function result = LH1(C,D)


pari = nozeronumber(C);
plus = 0;
if pari == 1
    result = log2(norm(D - C(:,1))); 
else
    result =log2(norm(D - C(:,pari)));
end

