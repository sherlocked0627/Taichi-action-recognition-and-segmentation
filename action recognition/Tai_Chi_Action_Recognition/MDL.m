function D = MDL(P)

len = nozeronumber(P);
C = zeros(3,len);
c = zeros(1,len);
C(:,1) = P(:,1);
c(1,1) = 1;
n = 1;
start = 1;
length = 1;
while (start + length < len)
    curr = start + length;
    cost1_1 = LH1(C,P(:,curr));
    cost1_2 = LDH(P,C,c,curr);
    cost1 = cost1_1 + cost1_2;
    cost2 = LH2(start,curr,P);
    if cost1 > cost2
        n = n + 1;
        C(:,n) = P(:,curr-1);
        c(1,n) = curr;
        start = curr - 1;
        length = 1;
    else
        length = length + 1;
    end
end
C(:,n+1) = P(:,len);
c(1,n+1) = len;

result = nozeronumber(C);


D = zeros(3,result-1);

F = zeros(3,result);
for i = 1:result-1
    D(1,i) = C(1,i+1) - C(1,i);
    D(2,i) = C(2,i+1) - C(2,i);
    D(3,i) = C(3,i+1) - C(3,i);
end
