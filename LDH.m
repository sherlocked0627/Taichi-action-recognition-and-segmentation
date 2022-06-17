function result = LDH(P,C,c,curr)

plus1 = 0;
plus2 = 0;

pari = nozeronumber(C);
D = P(:,curr);


for k = c(pari) : curr - 1
     plus1 = plus1 + D1(C(:,pari),D,P(:,k),P(:,k+1));
     plus2 = plus2 + D2(C(:,pari),D,P(:,k),P(:,k+1));
end
if plus1 > 0 && plus2 >0
     result = log2(plus1) + log2(plus2);
else
     result = 0;
end



